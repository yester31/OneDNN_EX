import torch # torch 1.9.0+cu111
import numpy as np
from compare import *

output_c = np.fromfile("C_Tensor", dtype=np.float32)
output_py = np.fromfile("py_0", dtype=np.float32)

if 0 :
    output_c = output_c.reshape((1,56,56,64))
    output_c = output_c.squeeze()
    output_c = output_c.transpose(2, 0, 1)
    output_c = output_c.flatten()

compare_two_tensor(output_py, output_c)