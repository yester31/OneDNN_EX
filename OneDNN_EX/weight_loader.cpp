#include <iostream>
#include <fstream>
#include <assert.h>
#include <string>
#include <iomanip>
#include <map>
#include "weight_loader.h"

std::map<std::string, Weights> loadWeights(std::string file)
{
	std::cout << "Loading weights: " << file << std::endl;
	std::map<std::string, Weights> weightMap;

	// Open weights file
	std::ifstream input(file);
	assert(input.is_open() && "Unable to load weight file.");

	// Read number of weight blobs
	int32_t count;
	input >> count;
	assert(count > 0 && "Invalid weight map file.");

	while (count--){
		uint32_t size;
		std::string name;
		input >> name >> std::dec >> size;
		std::vector<float> values(size);
		Weights wt{ values, size };
		uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
		for (uint32_t x = 0, y = size; x < y; ++x)
		{
			input >> std::hex >> val[x];
		}
		memcpy((wt.values).data(), (float*)val, sizeof(float) * size);
		//std::cout << "[layer name] : "<< std::left << std::setw(45) << name <<", [size] : " << size << std::endl;
		weightMap[name] = wt;
		
		free((void*)val);
	}

	return weightMap;
}


//int main() 
int weight_loader_test() 
{

	std::string file = "../model/resnet18.wts";
	std::map<std::string, Weights> wf = loadWeights(file);
	int len = wf["fc.bias"].count;
	for (int i = 0; i < len; i++){
		std::cout << (wf["fc.bias"].values)[i] << std::endl;
	}

	std::cout << "done!" << std::endl;

}