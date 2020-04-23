import numpy as np

import torch
import torch.nn as nn

torch.manual_seed(4)

model = nn.Linear(2, 4)

def get_weights():
    for i in model.parameters():
        print(i)

get_weights()

a = torch.rand([4, 2])

x = model(a)
# print(x)

# x.backward()

get_weights()