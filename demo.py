import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "deepseek-ai/deepseek-moe-16b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

def flatten_2d_list(twd_list):
    return [element for od_list in twd_list for element in od_list]

expert_num = 64
import numpy as np

def get_layer_output(module, input, output):
    expert_quant = flatten_2d_list(output[0])
    sample_nums = len(expert_quant)
    average_expert = sample_nums / expert_num
    expert_count_list = [expert_quant.count(i) for i in range(expert_num)]

    max_expert = np.argmax(expert_count_list)
    max_expert_tokens = expert_count_list[max_expert]
    gap = max_expert_tokens / average_expert

    print(gap)


text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=1)
layer_outputs = []
hooks = []
for decoder_layer in model.model.layers[1:-1]:
  hook = decoder_layer.mlp.gate.register_forward_hook(get_layer_output)
  break

result = model.generate(**inputs, max_new_tokens=1, temperature=0.0)
# full_expert_dict[idx] = layer_outputs

# for hook in hooks:
#     hook.remove()


# result = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(result)
# print(model)

"""
export TORCH_HOME=/scratch/zx22/zijie/cache
export HF_HOME=/scratch/zx22/zijie/cache
"""
