import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


model_id = "/vg_data/share/models/llama2-hf-converted/llama-2-7b"
lora_id = "/vg_data/share/models/llama2-hf-converted/fine_tuning_result/english_quotes/7b"
device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").to(device)


input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors='pt').to(device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("-----"*10)
model = PeftModel.from_pretrained(model, lora_id).to(device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


"""
CUDA_VISIBLE_DEVICES='0' python /data/ice/lt/llama-finetune/inference.py
"""