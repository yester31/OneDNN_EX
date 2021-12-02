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

// activation
memory activation_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	int mode)
{
	memory OUTPUT = memory(INPUT.get_desc(), engine);

	eltwise_forward::primitive_desc eltwise_pd;

	if (mode == 0) {//relu
		auto eltwise_d = eltwise_forward::desc(prop_kind::forward_inference, algorithm::eltwise_relu, INPUT.get_desc(), 0.f, 0.f);
		eltwise_pd = eltwise_forward::primitive_desc(eltwise_d, engine);
	}

	auto eltwise_prim = eltwise_forward(eltwise_pd);

	net.push_back(eltwise_prim);
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
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
	initTensor(inputs, inputs.size(), -5, 1);
	valueCheck(inputs, N, IC, IH, IW);
	tensor_dims t_dims{ N, IC, IH, IW };

	//[inputs]
	auto conv_src_md = memory::desc({ N, IC, IH, IW }, dt::f32, tag::nchw);
	auto user_src_memory = memory(conv_src_md, engine);
	write_to_dnnl_memory(inputs.data(), user_src_memory);

	memory output_s3 = activation_onednn(user_src_memory, net, net_args, engine, stream, 0);

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

