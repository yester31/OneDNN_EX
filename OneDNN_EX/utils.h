#pragma once
#include <io.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

void initTensor(std::vector<float> &output, uint64_t tot, float start = 0, float step = 1);

void valueCheck(std::vector<float> &data, int IN, int IC, int IH, int IW);

void tofile(std::vector<float> &Buffer, std::string fname = "../Validation_py/C_Tensor");