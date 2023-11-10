# demo for llama-finetune

## 当前状态
full_finetune.py提供了Trainer的重写，并且在alpaca数据集上进行了指令微调(PEQA论文的实验设计)
### 存在问题
device有bug，只能进入cuda：0

## 基本信息
目前使用了全量微调(full_finetune.py)和LoRA(llama_with_lora.py),对应执行脚本可见注释。

inference.py用于进行lora微调后的模型推理.

## 数据集使用

全量微调(full_finetune.py) 使用了Alpaca的52k数据, 数据集地址`/vg_data/share/dataset/alpaca`

LoRA(llama_with_lora.py) 使用了 `Abirate/english_quotes`(huggingface上的一个小数据集).


## Reference

全量微调：https://github.com/tatsu-lab/stanford_alpaca
PEQA:<<Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization>>