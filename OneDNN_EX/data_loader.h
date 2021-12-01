#pragma once
#include <io.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "opencv2/opencv.hpp"

int SearchFile(const std::string& folder_path, std::vector<std::string> &file_names, bool recursive = false);

int preprocess(std::vector<float> &output, std::vector<uint8_t> &input, int IN, int IC, int IH, int IW);