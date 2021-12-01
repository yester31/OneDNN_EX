#include "data_loader.h"
#include "utils.h"

//파일 이름 가져오기(DFS) window용
int SearchFile(const std::string& folder_path, std::vector<std::string> &file_names, bool recursive){
	_finddata_t file_info;
	std::string any_file_pattern = folder_path + "\\*";
	intptr_t handle = _findfirst(any_file_pattern.c_str(), &file_info);
	if (handle == -1){
		std::cerr << "folder path not exist: " << folder_path << std::endl;
		return -1;
	}
	do{
		std::string file_name = file_info.name;
		if (recursive) {
			if (file_info.attrib & _A_SUBDIR) {//check whtether it is a sub direcotry or a file
				if (file_name != "." && file_name != ".."){
					std::string sub_folder_path = folder_path + "//" + file_name;
					SearchFile(sub_folder_path, file_names);
					std::cout << "a sub_folder path: " << sub_folder_path << std::endl;
				}
			}
			else{
				std::string file_path = folder_path + "/" + file_name;
				file_names.push_back(file_path);
			}
		}
		else {
			if (!(file_info.attrib & _A_SUBDIR)) {//check whtether it is a sub direcotry or a file
				std::string file_path = folder_path + "/" + file_name;
				file_names.push_back(file_path);
			}
		}
	} while (_findnext(handle, &file_info) == 0);
	_findclose(handle);
	return 0;
}

// 전처리 : BGR -> RGB, NHWC->NCHW, Normalize (0 ~ 1)
int preprocess(std::vector<float> &output, std::vector<uint8_t> &input, int IN, int IC, int IH, int IW) {

	int C_offset, H_offset, W_offset, g_in, g_out;
	int N_offset = IH * IW * IC;
	for (int ⁠n_idx = 0; ⁠n_idx < IN; ⁠n_idx++) {
		H_offset = ⁠n_idx * N_offset;
		for (int ⁠h_idx = 0; ⁠h_idx < IH; ⁠h_idx++) {
			W_offset = ⁠h_idx * IW * IC + H_offset;
			for (int w_idx = 0; w_idx < IW; w_idx++) {
				C_offset = w_idx * IC + W_offset;
				for (int ⁠c_idx = 0; ⁠c_idx < IC; ⁠c_idx++) {
					g_in = C_offset + 2 - ⁠c_idx;
					g_out = H_offset + ⁠c_idx * IH * IW + ⁠h_idx * IW + w_idx;
					output[g_out] = static_cast <float>(input[g_in]) / 255.f;
				}
			}
		}
	}
}


//int main()
int test_dataloader()
{
	// 0. 이미지경로 로드
	std::string img_dir = "../data";
	std::vector<std::string> file_names;
	if (SearchFile(img_dir.c_str(), file_names) < 0) {
		std::cerr << "data search error" << std::endl;
		return 0;
	}

	// 1. 이미지 데이터 로드
	int batch_size = 1;
	int input_width = 224;
	int input_height = 224;
	cv::Mat img(input_height, input_width, CV_8UC3);
	cv::Mat ori_img;
	std::vector<uint8_t> input(batch_size * input_height * input_width * 3);

	for (int idx = 0; idx < batch_size; idx++) {
		cv::Mat ori_img = cv::imread(file_names[idx]);
		cv::resize(ori_img, img, img.size(), cv::INTER_LINEAR);
		int offset = idx * input_height * input_width * 3;
		memcpy(input.data() + offset, img.data, input_height * input_width * 3);
	}

	std::vector<float> output(input.size());
	
	preprocess(output, input, batch_size, 3, input_height, input_width);
	tofile(output);

	return 0;
}