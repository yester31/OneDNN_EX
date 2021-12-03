import torch # torch 1.9.0+cu111
import numpy as np
from compare import *
import torch.nn.functional as F
OC = 3
IN = 1
IC = 3
IH = 4
IW = 4
KH = 2
KW = 2
SH = 2
SW = 2
TP = 0
BP = 0
LP = 0
RP = 0


# OC = 64
# IN = 1
# IC = 3
# IH = 224
# IW = 224
# KH = 7
# KW = 7
# SH = 2
# SW = 2
# TP = 3
# BP = 3
# LP = 3
# RP = 3

with torch.no_grad():
    #weight = torch.ones([OC, IC, KH, KW], dtype=torch.float32, requires_grad=False)
    weight = np.arange(0, OC* IC* KH* KW).reshape(OC, IC, KH, KW)
    weight = torch.from_numpy(weight).type(torch.FloatTensor)
    #print(weight)

    bias = torch.ones([OC], dtype=torch.float32, requires_grad=False)
    #print(bias)

    input_np = np.arange(0, IN * IC * IH * IW).reshape(IN, IC, IH, IW)
    input = torch.from_numpy(input_np).type(torch.FloatTensor)
    #print(input)

    p2d = (LP, RP, TP, BP)
    input_padded = torch.nn.functional.pad(input, p2d, "constant", 0)
    #print(input_padded)

    convolution = torch.nn.Conv2d(IC, OC, (KH, KH), stride=(SH, SW), bias=False)
    convolution.weight = torch.nn.Parameter(weight)
    #conservertive_convolution.bias = torch.nn.Parameter(bias)
    output = convolution(input_padded)
    #print(output)

    batchnorm = torch.nn.BatchNorm2d(OC, eps=1e-5)
    batchnorm.weight = torch.nn.Parameter(bias)
    batchnorm.bias = torch.nn.Parameter(bias)
    batchnorm.running_mean.data = torch.nn.Parameter(bias)
    batchnorm.running_var.data = torch.nn.Parameter(bias)
    batchnorm.eval()
    output2 = batchnorm(output)
    #print(output2)

    relu = torch.nn.ReLU()
    output2 = relu(output2)
    #print(output2)

    maxpooling = torch.nn.MaxPool2d(2, stride=2)
    output3 = maxpooling(output2)
    #print(output3)

    output4 = F.avg_pool2d(output3, (output3.shape[2], output3.shape[3]))
    #print(output4)

    output4 = torch.flatten(output4, 1)

    weight2 = torch.ones([OC*10], dtype=torch.float32, requires_grad=False).reshape(10, OC)
    bias2 = torch.ones([10], dtype=torch.float32, requires_grad=False)

    fc = torch.nn.Linear(OC, 10)
    fc.weight = torch.nn.Parameter(weight2)
    fc.bias = torch.nn.Parameter(bias2)
    output5 = fc(output4)
    # print(output5)


py_data = output
#py_data = output2

output_c = np.fromfile("C_Tensor", dtype=np.float32)
output_c = output_c.reshape((py_data.shape[0],py_data.shape[2],py_data.shape[3],py_data.shape[1]))
output_c = output_c.squeeze()
output_c = output_c.transpose(2, 0, 1)
output_c = output_c.flatten()
output_py = py_data.detach().numpy().flatten()

compare_two_tensor2(output_py, output_c)