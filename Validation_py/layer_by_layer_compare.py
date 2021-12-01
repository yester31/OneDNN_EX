import torch # torch 1.9.0+cu111
import numpy as np
from compare import *

output_c = np.fromfile("C_Tensor", dtype=np.float32)
output_py = np.fromfile("py_0", dtype=np.float32)

compare_two_tensor(output_py, output_c)