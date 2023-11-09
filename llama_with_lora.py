import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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


os.environ["WANDB_DISABLED"]="true"
model_id = "/vg_data/share/models/llama2-hf-converted/llama-2-7b"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

# >>> The custom quantization method is executed here！！！
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

#  For that use the `prepare_model_for_kbit_training` method from PEFT.
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


# This configuration is for llama-2, in particular the target_modules
config = LoraConfig(
    r=8, # dimension of the updated matrices
    lora_alpha=32, # parameter for scaling
    target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
    lora_dropout=0.1, # dropout probability for layers
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model,config)
print_trainable_parameters(model)

# Let's load a common dataset, english quotes, to fine tune our model on famous quotes.
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# Run the cell below to run the training!

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
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

trainer.save_model("/vg_data/share/models/llama2-hf-converted/fine_tuning_result/english_quotes/7b")

"""
CUDA_VISIBLE_DEVICES='2' python /data/ice/lt/llama-finetune/llama_with_lora.py
"""