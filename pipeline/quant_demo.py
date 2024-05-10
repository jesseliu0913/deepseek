import torch
import torch.nn as nn

import bitsandbytes as bnb
from bitsandbytes.nn import Linear8bitLt

fp16_model = nn.Sequential(
    nn.Linear(64, 64),
    nn.Linear(64, 64)
)

int8_model = nn.Sequential(
    Linear8bitLt(64, 64, has_fp16_weights=False),
    Linear8bitLt(64, 64, has_fp16_weights=False)
)

print(fp16_model.state_dict())
int8_model.load_state_dict(fp16_model.state_dict())
int8_model = int8_model.to(0) # Quantization happens here
print(int8_model.state_dict())
