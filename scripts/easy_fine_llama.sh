CUDA_VISIBLE_DEVICES=1 \
python3 /data/ice/lt/llama-finetune/easy_finetune_llama.py \
--save_path "/vg_data/share/models/llama2-hf-converted/fine_tuning_result/english_quotes/test" \
--output_dir "outputs" \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--warmup_steps 2 \
--save_steps 1000 \
--max_steps 10 \
--learning_rate 2e-4 \
--fp16 True \
--logging_steps 1 \
--optim "paged_adamw_32bit" \
--report_to none 