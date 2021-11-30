import torch # torch 1.9.0+cu111
import numpy as np
from compare import *

OC = 3
IN = 1
IC = 1
IH = 4
IW = 4
KH = 3
KW = 3
TP = 1
BP = 1
LP = 1
RP = 1
with torch.no_grad():
    weight = torch.ones([OC, IC, KH, KW], dtype=torch.float32, requires_grad=False)
    print(weight)

    bias = torch.ones([OC], dtype=torch.float32, requires_grad=False)*2
    print(bias)

    input_np = np.arange(-5, IN * IC * IH * IW - 5).reshape(IN, IC, IH, IW)
    input = torch.from_numpy(input_np).type(torch.FloatTensor)
    print(input)

    p2d = (LP, RP, TP, BP)
    input_padded = torch.nn.functional.pad(input, p2d, "constant", 0)
    print(input_padded)

    convolution = torch.nn.Conv2d(IC, OC, (KH, KH), stride=(1, 1), bias=False)
    convolution.weight = torch.nn.Parameter(weight)
    #conservertive_convolution.bias = torch.nn.Parameter(bias)
    output = convolution(input_padded)
    print(output)

    batchnorm = torch.nn.BatchNorm2d(OC, eps=1e-5)
    batchnorm.weight = torch.nn.Parameter(bias)
    batchnorm.bias = torch.nn.Parameter(bias)
    batchnorm.running_mean.data = torch.nn.Parameter(bias)
    batchnorm.running_var.data = torch.nn.Parameter(bias)
    batchnorm.eval()
    output2 = batchnorm(output)
    print(output2)

    maxpooling = torch.nn.MaxPool2d(2, stride=2)
    output3 = maxpooling(output2)
    print(output3)

    gap = torch.nn.AvgPool2d(2,stride=1)
    output4 = gap(output3)
    print(output4)

    output4 = torch.flatten(output4, 1)

    weight2 = torch.ones([OC*10], dtype=torch.float32, requires_grad=False).reshape(10, OC)
    bias2 = torch.ones([10], dtype=torch.float32, requires_grad=False)

    fc = torch.nn.Linear(OC, 10)
    fc.weight = torch.nn.Parameter(weight2)
    fc.bias = torch.nn.Parameter(bias2)
    output5 = fc(output4)
    print(output5)


# input_padded2 = torch.nn.functional.pad(output, p2d, "constant", 0)
# output2 = conservertive_convolution(input_padded2)
# print(output2)

# output_c = np.fromfile("../output/C_Tensor_zp", dtype=np.float32)
# output_py = output.detach().numpy().flatten()
#
# compare_two_tensor(output_py, output_c)