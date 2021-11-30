#pragma once

void initTensor(std::vector<float> &output, uint64_t tot, float start = 0, float step = 1);
void valueCheck(std::vector<float> &data, int IN, int IC, int IH, int IW);