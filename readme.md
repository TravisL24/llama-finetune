# demo for llama-finetune


目前使用了全量微调(full_finetune.py)和LoRA(llama_with_lora.py),对应执行脚本可见注释。

inference.py用于进行lora微调后的模型推理，全量微调还在尝试使用deepspeed解决OOM问题。

## 数据集使用

全量微调(full_finetune.py) 使用了Alpaca的52k数据, 数据集地址`/vg_data/share/dataset/alpaca`

LoRA(llama_with_lora.py) 使用了 `Abirate/english_quotes`(huggingface上的一个小数据集).


## Reference

全量微调：https://github.com/tatsu-lab/stanford_alpaca