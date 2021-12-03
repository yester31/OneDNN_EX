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

memory conv2d_onednn_wo_bias_old(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights, tensor_dims &t_dims, int OC, int KH, int KW, int SH, int SW, int TP, int BP, int LP, int RP, int Acti);

memory conv2d_onednn_wo_bias(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights, tensor_dims &t_dims, int OC, int KH, int KW, int SH, int SW, int TP, int BP, int LP, int RP, int Acti);

void simple_net(engine::kind engine_kind, int times = 100) {

	//[Initialize engine and stream]
	engine engine(engine_kind, 0);
	stream stream(engine);

	//[Create network]
	std::vector<primitive> net;
	std::vector<std::unordered_map<int, memory>> net_args;

	//int OC = 64;
	//int N = 1;
	//int IC = 3;
	//int IH = 224;
	//int IW = 224;
	//int KH = 7;
	//int KW = 7;
	//int SH = 2;
	//int SW = 2;
	//int TP = 3;
	//int BP = 3;
	//int LP = 3;
	//int RP = 3;

	int OC = 3;
	int N = 1;
	int IC = 3;
	int IH = 4;
	int IW = 4;
	int KH = 2;
	int KW = 2;
	int SH = 2;
	int SW = 2;
	int TP = 0;
	int BP = 0;
	int LP = 0;
	int RP = 0;

	// weight[OC][lC][KH][KW] 
	// 임시 weight 값 
	std::vector<float> weights(OC * IC * KH * KW);
	initTensor(weights, weights.size(), 0, 1);
	valueCheck(weights, OC, IC, KH, KW);

	// d[N][IC][IH][IW] 
	// 임시 input 값 
	std::vector<float> inputs(N * IC * IH * IW);
	initTensor(inputs, inputs.size(), 0, 1);
	valueCheck(inputs, N, IC, IH, IW);

	//[inputs]
	auto conv_src_md = memory::desc({ N, IC, IH, IW }, dt::f32, tag::nchw);
	auto user_src_memory = memory(conv_src_md, engine);
	write_to_dnnl_memory(inputs.data(), user_src_memory);

	tensor_dims t_dims{ N, IC, IH, IW };

	memory output_s = conv2d_onednn_wo_bias(
		user_src_memory,
		net, net_args, engine, stream,
		weights,
		t_dims, OC, KH, KW, SH, SW, TP, BP, LP, RP, 0);


	//[Execute model]
	auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

	for (int j = 0; j < 100; ++j) {
		assert(net.size() == net_args.size() && "something is missing");
		for (size_t i = 0; i < net.size(); ++i)
			net.at(i).execute(stream, net_args.at(i));
	}
	stream.wait();
	auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

	std::vector<float> outputs(t_dims.N * t_dims.IC* t_dims.IH* t_dims.IW);
	read_from_dnnl_memory(outputs.data(), net_args.at(net.size() - 1).find(DNNL_ARG_DST)->second);
	valueCheck(outputs, t_dims.N , t_dims.IC, t_dims.IH, t_dims.IW);
	tofile(outputs);
	std::cout << "Use time: " << (end - begin) << " ms per iteration." << std::endl;
	std::cout << "done!!!" << std::endl;
	std::cout << "layer count : " << net.size() << std::endl;
	// old 174 161 147 160 250 
	// new 56 70 50 140 46
}

void cnn_inference_f32(engine::kind engine_kind) {
	int times = 100;
	simple_net(engine_kind, times);
}

int main(int argc, char **argv) {
	return handle_example_errors(cnn_inference_f32, parse_engine_kind(argc, argv));
}

memory conv2d_onednn_wo_bias_old(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights, tensor_dims &t_dims, int OC, int KH, int KW, int SH, int SW, int TP, int BP, int LP, int RP, int Acti)
{
	int OH = (t_dims.IH - KH + TP + BP) / SH + 1; // output height
	int OW = (t_dims.IW - KW + LP + RP) / SW + 1; // output width

	auto conv_dst_md = memory::desc({ t_dims.N, OC, OH, OW }, dt::f32, tag::nchw);
	auto OUTPUT = memory(conv_dst_md, engine);

	auto conv_weights_md = memory::desc({ OC, t_dims.IC, KH, KW }, dt::f32, tag::oihw);
	auto user_weights_mem = memory(conv_weights_md, engine);
	write_to_dnnl_memory(weights.data(), user_weights_mem);

	// Create operation descriptor.
	auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_auto, INPUT.get_desc(), conv_weights_md,
		conv_dst_md, { SH, SW }, { TP, LP }, { BP, RP });

	// Activation func
	convolution_forward::primitive_desc conv_pd;
	if (Acti == 1) {
		// Create primitive post-ops (ReLU).
		const float scale = 1.f;
		const float alpha = 0.f;
		const float beta = 0.f;
		post_ops conv_ops;
		conv_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
		primitive_attr conv_attr;
		conv_attr.set_post_ops(conv_ops);
		conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, engine);
	}
	else { // linear
		conv_pd = convolution_forward::primitive_desc(conv_desc, engine);
	}

	// Create the primitive.
	net.push_back(convolution_forward(conv_pd));

	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_WEIGHTS, user_weights_mem },
		{ DNNL_ARG_DST, OUTPUT }
		});

	t_dims.IC = OC;
	t_dims.IH = OH;
	t_dims.IW = OW;
	return OUTPUT;
}

memory conv2d_onednn_wo_bias(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights, tensor_dims &t_dims, int OC, int KH, int KW, int SH, int SW, int TP, int BP, int LP, int RP, int Acti)
{
	int OH = (t_dims.IH - KH + TP + BP) / SH + 1; // output height
	int OW = (t_dims.IW - KW + LP + RP) / SW + 1; // output width

	memory::dims conv2_src_tz = { t_dims.N, t_dims.IC, t_dims.IH, t_dims.IW };
	memory::dims conv2_weights_tz = { OC, t_dims.IC, KH, KW };
	memory::dims conv2_dst_tz = { t_dims.N, OC, OH, OW };
	memory::dims conv2_strides = { SH, SW };
	memory::dims conv2_padding1 = { TP, LP };
	memory::dims conv2_padding2 = { BP, RP };

	// create memory for user data
	auto conv2_user_weights_memory = memory({ {conv2_weights_tz}, dt::f32, tag::oihw }, engine);
	write_to_dnnl_memory(weights.data(), conv2_user_weights_memory);

	// create memory descriptors for convolution data w/ no specified format
	auto conv2_src_md = memory::desc({ conv2_src_tz }, dt::f32, tag::any);
	auto conv2_weights_md = memory::desc({ conv2_weights_tz }, dt::f32, tag::any);
	auto conv2_dst_md = memory::desc({ conv2_dst_tz }, dt::f32, tag::any);

	// create a convolution
	auto conv2_desc = convolution_forward::desc(prop_kind::forward_inference, algorithm::convolution_direct,
		conv2_src_md, conv2_weights_md, conv2_dst_md, conv2_strides, conv2_padding1, conv2_padding2);

	// Activation func
	convolution_forward::primitive_desc conv2_prim_desc;

	if (Acti == 1) {
		// Create primitive post-ops (ReLU).
		const float scale = 1.f;
		const float alpha = 0.f;
		const float beta = 0.f;
		post_ops conv_ops;
		conv_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
		primitive_attr conv_attr;
		conv_attr.set_post_ops(conv_ops);

		// Create primitive descriptor.
		conv2_prim_desc = convolution_forward::primitive_desc(conv2_desc, conv_attr, engine);
	}
	else { // linear
		conv2_prim_desc = convolution_forward::primitive_desc(conv2_desc, engine);
	}

	auto conv2_src_memory = INPUT;
	if (conv2_prim_desc.src_desc() != INPUT.get_desc()) {
		conv2_src_memory = memory(conv2_prim_desc.src_desc(), engine);
		reorder(INPUT, conv2_src_memory).execute(engine_stream, { {DNNL_ARG_FROM, INPUT},{DNNL_ARG_TO, conv2_src_memory} });
		engine_stream.wait();
	}
	auto conv2_weights_memory = conv2_user_weights_memory;
	if (conv2_prim_desc.weights_desc() != conv2_user_weights_memory.get_desc()) {
		conv2_weights_memory = memory(conv2_prim_desc.weights_desc(), engine);
		reorder(conv2_user_weights_memory, conv2_weights_memory).execute(engine_stream, { {DNNL_ARG_FROM, conv2_user_weights_memory},{DNNL_ARG_TO, conv2_weights_memory} });
		engine_stream.wait();
	}

	auto conv2_dst_memory = memory(conv2_prim_desc.dst_desc(), engine);

	net.push_back(convolution_forward(conv2_prim_desc));
	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, conv2_src_memory },
		{ DNNL_ARG_WEIGHTS, conv2_weights_memory },
		{ DNNL_ARG_DST, conv2_dst_memory }
		});

	t_dims.IC = OC;
	t_dims.IH = OH;
	t_dims.IW = OW;
	return conv2_dst_memory;
}

