#pragma once
#include <map>
#include <vector>

class Weights
{
public:
	std::vector<float> values; 
	int64_t count;      //!< The number of weights in the array.
};

std::map<std::string, Weights> loadWeights(std::string file);