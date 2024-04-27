import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "deepseek-ai/deepseek-moe-16b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

def flatten_2d_list(twd_list):
    return [element for od_list in twd_list for element in od_list]

expert_num = 64

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
model_raw_dict = model.state_dict()
layer_outputs = []
hooks = []
set_layer = 1  # 1-27
set_expert = 0  # 0-64
for layer_index, decoder_layer in enumerate(model.model.layers[1:-1]):
  if set_layer == layer_index:
    decoder_layer.mlp.gate.n_routed_experts = 63

model_new_dict = model.state_dict()

for key in list(model_raw_dict.keys()):
    model_new_dict[key] = model_raw_dict[key]

for key in list(model_raw_dict.keys()):
    if 'gate' in key:
      print(key)



# result = model.generate(**inputs, max_new_tokens=1, temperature=0.0)
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
