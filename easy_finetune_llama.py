import os
import time
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from transformers.trainer_utils import TrainOutput, speed_metrics, find_executable_batch_size
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset




def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


###########################
#  overwriting Trainer    #
###########################
class ModifiedTrianer(Trainer):

    def train(
        self,
        **kwargs,
    ):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        self._train_batch_size = self.args.train_batch_size

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )

        return inner_training_loop(args=args)


    # The whole triaing loop
    def _inner_training_loop(
       self, batch_size=None, 
       args=None, 
    ):
        self._train_batch_size = batch_size
        print(f"Currently training with a batch size of: {self._train_batch_size}")

        # Setting up some training control variables
        train_dataloader = self.get_train_dataloader()
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        num_examples = self.num_examples(train_dataloader)

        max_steps = args.max_steps
        num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
            args.max_steps % num_update_steps_per_epoch > 0
        )
        num_train_samples = args.max_steps * total_train_batch_size
        if args.include_tokens_per_second:
            num_train_tokens = (
                self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
            )

        # create_optimizer_and_scheduler. Still can overwriting
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)


        # Train!
        print("***** Running training *****")
        print(f"  Num examples = {num_examples:,}")
        print(f"  Num Epochs = {num_train_epochs:,}")
        print(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {max_steps:,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        total_batched_samples = 0
        # >>> here is loop <<<
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            # self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            steps_skipped = 0

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # with self.accelerator.accumulate(model):
                #     tr_loss_step = self.training_step(model, inputs)

                tr_loss_step = self.training_step(model, inputs)

                tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs)) # the number of float-point ops

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # >>> Other gradient modification methods can be performed here <<<
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                        )

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        print("\n\nTraining completed.")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    def training_step(self, model, inputs):
        # >>> Overwriting the training method here. One step <<<
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)

        loss.backward()


        # >>> Operate on gradient here <<<
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         if param.grad is not None:
        #             print(f"{name}, gradient: {param.grad.mean()}")
        #         else:
        #             print(f"{name} has not gradient")
        
        return loss.detach() / self.args.gradient_accumulation_steps


    # def compute_loss(self, model, inputs, return_outputs=False):
    #     return model(
    #         input_ids=inputs["inputs_id"],
    #         attention_mask=torch.ones_like(inputs["inputs_id"]),
    #         labels=inputs["inputs_id"],
    #     ).loss



os.environ["WANDB_DISABLED"]="true"

###########################################
#     load dataset tokenizer and model    #
###########################################

model_id = "/vg_data/share/models/llama2-hf-converted/llama-2-7b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").half()

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)


model.gradient_checkpointing_enable()
print_trainable_parameters(model)

####################################
#  set up tokenizer and trainer    #
####################################
tokenizer.pad_token = tokenizer.eos_token

trainer = ModifiedTrianer(
    model=model,
    train_dataset=data["train"], 
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        save_steps=1000,        
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit", # default = adamw_torch_fused
        report_to="none" # turn wandb off
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

trainer.save_model("/vg_data/share/models/llama2-hf-converted/fine_tuning_result/english_quotes/test")

"""
CUDA_VISIBLE_DEVICES='1' python /data/ice/lt/llama-finetune/easy_finetune_llama.py
"""