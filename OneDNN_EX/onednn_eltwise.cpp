#include <assert.h>
#include <chrono>
#include <vector>
#include <unordered_map>
#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"
#include <iomanip>
#include "utils.h"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

//
memory eltwise_onednn(memory &INPUT, memory &INPUT2, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream)
{
	memory OUTPUT = memory(INPUT.get_desc(), engine);

	// Create primitive descriptor.
	auto sum_pd = sum::primitive_desc({ 1, 1 }, { INPUT.get_desc() , INPUT2.get_desc() }, engine);

	// Create the primitive.
	auto sum_prim = sum(sum_pd);

	net.push_back(sum_prim);
	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_MULTIPLE_SRC + 0, INPUT },
		{ DNNL_ARG_MULTIPLE_SRC + 1, INPUT2 },
		{ DNNL_ARG_DST, OUTPUT }
		});

	return OUTPUT;
}



void simple_net(engine::kind engine_kind, int times = 100) {

	//[Initialize engine and stream]
	engine engine(engine_kind, 0);
	stream stream(engine);

	//[Create network]
	std::vector<primitive> net;
	std::vector<std::unordered_map<int, memory>> net_args;

	int N = 1;
	int IC = 1;
	int IH = 4;
	int IW = 4;

	// d[IN][IC][IH][IW] 
	// ÀÓ½Ã input °ª 
	std::vector<float> inputs(N * IC * IH * IW);
	initTensor(inputs, inputs.size(), 5, 1);
	valueCheck(inputs, N, IC, IH, IW);

	std::vector<float> inputs2(N * IC * IH * IW);
	initTensor(inputs2, inputs2.size(), 5, 1);
	valueCheck(inputs2, N, IC, IH, IW);

	tensor_dims t_dims{ N, IC, IH, IW };

	//[inputs]
	auto conv_src_md = memory::desc({ N, IC, IH, IW }, dt::f32, tag::nchw);
	auto user_src_memory = memory(conv_src_md, engine);
	write_to_dnnl_memory(inputs.data(), user_src_memory);

	auto conv_src_md2 = memory::desc({ N, IC, IH, IW }, dt::f32, tag::nchw);
	auto user_src_memory2 = memory(conv_src_md2, engine);
	write_to_dnnl_memory(inputs2.data(), user_src_memory2);

	memory output_s3 = eltwise_onednn(user_src_memory, user_src_memory2, net, net_args, engine, stream);

	//[Execute model]
	for (int j = 0; j < 1; ++j) {
		assert(net.size() == net_args.size() && "something is missing");
		for (size_t i = 0; i < net.size(); ++i)
			net.at(i).execute(stream, net_args.at(i));
	}

	stream.wait();
	//std::vector<float> outputs2(N * OC * OH * OW);
	//read_from_dnnl_memory(outputs2.data(), net_args.at(net.size() - 2).find(DNNL_ARG_DST)->second);
	//valueCheck(outputs2, N, OC, OH, OW);
	std::vector<float> outputs(t_dims.N * t_dims.IC * t_dims.IH * t_dims.IW);
	read_from_dnnl_memory(outputs.data(), net_args.at(net.size() - 1).find(DNNL_ARG_DST)->second);
	valueCheck(outputs, t_dims.N, t_dims.IC, t_dims.IH, t_dims.IW);
	std::cout << "done!!!" << std::endl;
	std::cout << "layer count : " << net.size() << std::endl;
}

void cnn_inference_f32(engine::kind engine_kind) {
	auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	int times = 100;
	simple_net(engine_kind, times);
	auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	std::cout << "Use time: " << (end - begin) << " ms per iteration." << std::endl;
}

int main(int argc, char **argv) {
	return handle_example_errors(cnn_inference_f32, parse_engine_kind(argc, argv));
}

