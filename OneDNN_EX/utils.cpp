#include <iostream>
#include <iomanip>
#include <vector>

void initTensor(std::vector<float> &output, uint64_t tot, float start, float step)
{
	std::cout << "===== InitTensor func (scalar or step)=====" << std::endl;
	float count = start;
	for (int i = 0; i < tot; i++) {
		output[i] = count;
		count += step;
	}
}

void valueCheck(std::vector<float> &data, int IN, int IC, int IH, int IW) {
	std::cout << "===== valueCheck func =====" << std::endl;
	int N_offset = IC * IH * IW;
	int C_offset, H_offset, W_offset, g_idx;
	for (int ⁠n_idx = 0; ⁠n_idx < IN; ⁠n_idx++) {
		C_offset = ⁠n_idx * N_offset;
		for (int ⁠c_idx = 0; ⁠c_idx < IC; ⁠c_idx++) {
			H_offset = ⁠c_idx * IW * IH + C_offset;
			for (int ⁠h_idx = 0; ⁠h_idx < IH; ⁠h_idx++) {
				W_offset = ⁠h_idx * IW + H_offset;
				for (int w_idx = 0; w_idx < IW; w_idx++) {
					g_idx = w_idx + W_offset;
					std::cout << std::setw(5) << data[g_idx] << " ";
				}std::cout << std::endl;
			}std::cout << std::endl; std::cout << std::endl;
		}
	}
}