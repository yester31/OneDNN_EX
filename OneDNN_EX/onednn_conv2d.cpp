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

void conv2d_onednn(
	memory &OUTPUT, memory &INPUT,
	std::vector<float> &weights,
	std::vector<float> &bias,
	std::vector<primitive> &net,
	std::vector<std::unordered_map<int, memory>> &net_args,
	engine &engine, stream &engine_stream,
	int N, int IC, int IH, int IW, int OC,
	int KH, int KW, int	SH, int SW,
	int TP, int BP, int LP, int RP,
	int Acti = 0)
{
	int OH = (IH - KH + TP + BP) / SH + 1; // output height
	int OW = (IW - KW + LP + RP) / SW + 1; // output width

	auto conv_dst_md = memory::desc({ N, OC, OH, OW }, dt::f32, tag::nchw);
	OUTPUT = memory(conv_dst_md, engine);

	auto conv_weights_md = memory::desc({ OC, IC, KH, KW }, dt::f32, tag::oihw);
	auto user_weights_mem = memory(conv_weights_md, engine);
	write_to_dnnl_memory(weights.data(), user_weights_mem);

	auto user_bias_md = memory::desc({ OC }, dt::f32, tag::a);
	auto user_bias_mem = memory(user_bias_md, engine);
	write_to_dnnl_memory(bias.data(), user_bias_mem);

	// Create operation descriptor.
	auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_auto, INPUT.get_desc(), conv_weights_md,
		user_bias_md, conv_dst_md, { SH, SW }, { TP, LP }, { BP, RP });

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

		// Create primitive descriptor.
		conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, engine);
	}
	else { // linear
		conv_pd = convolution_forward::primitive_desc(conv_desc, engine);
	}

	// Create the primitive.
	auto conv_prim = convolution_forward(conv_pd);
	net.push_back(conv_prim);

	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_WEIGHTS, user_weights_mem },
		{ DNNL_ARG_BIAS, user_bias_mem },
		{ DNNL_ARG_DST, OUTPUT }
		});

}

void conv2d_onednn_wo_bias(
	memory &OUTPUT, memory &INPUT,
	std::vector<float> &weights,
	std::vector<primitive> &net,
	std::vector<std::unordered_map<int, memory>> &net_args,
	engine &engine, stream &engine_stream,
	int N, int IC, int IH, int IW, int OC,
	int KH, int KW, int	SH, int SW,
	int TP, int BP, int LP, int RP,
	int Acti = 0)
{
	int OH = (IH - KH + TP + BP) / SH + 1; // output height
	int OW = (IW - KW + LP + RP) / SW + 1; // output width

	auto conv_dst_md = memory::desc({ N, OC, OH, OW }, dt::f32, tag::nchw);
	OUTPUT = memory(conv_dst_md, engine);

	auto conv_weights_md = memory::desc({ OC, IC, KH, KW }, dt::f32, tag::oihw);
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

		// Create primitive descriptor.
		conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, engine);
	}
	else { // linear
		conv_pd = convolution_forward::primitive_desc(conv_desc, engine);
	}

	// Create the primitive.
	auto conv_prim = convolution_forward(conv_pd);
	net.push_back(conv_prim);

	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_WEIGHTS, user_weights_mem },
		{ DNNL_ARG_DST, OUTPUT }
		});

}

void conv2d_onednn_wo_bias2(
	memory &OUTPUT, memory &INPUT,
	std::vector<float> &weights,
	std::vector<primitive> &net,
	std::vector<std::unordered_map<int, memory>> &net_args,
	engine &engine, stream &engine_stream,
	int N, int IC, int IH, int IW, int OC,
	int KH, int KW, int	SH, int SW,
	int TP, int BP, int LP, int RP,
	int Acti = 0)
{
	int OH = (IH - KH + TP + BP) / SH + 1; // output height
	int OW = (IW - KW + LP + RP) / SW + 1; // output width

	auto conv_dst_md = memory::desc({ N, OC, OH, OW }, dt::f32, tag::nchw);
	OUTPUT = memory(conv_dst_md, engine);

	auto conv_weights_md = memory::desc({ OC, IC, KH, KW }, dt::f32, tag::oihw);
	auto user_weights_mem = memory(conv_weights_md, engine);
	write_to_dnnl_memory(weights.data(), user_weights_mem);

	auto conv_src_md2 = memory::desc({ N, IC, IH, IW }, dt::f32, tag::any);
	auto conv_weights_md2 = memory::desc({ OC, IC, KH, KW }, dt::f32, tag::any);
	auto conv_dst_md2 = memory::desc({ N, OC, OH, OW }, dt::f32, tag::any);

	// Create operation descriptor.
	auto conv_desc = convolution_forward::desc(prop_kind::forward_inference, algorithm::convolution_auto,
		conv_src_md2, conv_weights_md2, conv_dst_md2, { SH, SW }, { TP, LP }, { BP, RP });

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

		// Create primitive descriptor.
		conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, engine);
	}
	else { // linear
		conv_pd = convolution_forward::primitive_desc(conv_desc, engine);
	}

	auto conv_src_mem = INPUT;
	auto conv_weights_mem = user_weights_mem;
	auto conv_dst_mem = OUTPUT;

	if (conv_pd.src_desc() != INPUT.get_desc()) {
		conv_src_mem = memory(conv_pd.src_desc(), engine);

		reorder(INPUT, conv_src_mem).execute(engine_stream, INPUT, conv_src_mem);
		engine_stream.wait();
	}

	if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
		conv_weights_mem = memory(conv_pd.weights_desc(), engine);

		reorder(user_weights_mem, conv_weights_mem).execute(engine_stream, user_weights_mem, conv_weights_mem);
		engine_stream.wait();
	}

	if (conv_pd.dst_desc() != OUTPUT.get_desc()) {
		conv_dst_mem = memory(conv_pd.dst_desc(), engine);
	}

	// Create the primitive.
	auto conv_prim = convolution_forward(conv_pd);
	net.push_back(conv_prim);

	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_WEIGHTS, user_weights_mem },
		{ DNNL_ARG_DST, OUTPUT }
		});

}


void simple_net(engine::kind engine_kind, int times = 100) {

	//[Initialize engine and stream]
	engine engine(engine_kind, 0);
	stream stream(engine);

	//[Create network]
	std::vector<primitive> net;
	std::vector<std::unordered_map<int, memory>> net_args;

	int OC = 1;
	int N = 1;
	int IC = 1;
	int IH = 4;
	int IW = 4;
	int KH = 3;
	int KW = 3;
	int SH = 1;
	int SW = 1;
	int TP = 1;
	int BP = 1;
	int LP = 1;
	int RP = 1;

	// weight[OC][lC][KH][KW] 
	// 임시 weight 값 
	std::vector<float> weights(OC * IC * KH * KW);
	initTensor(weights, weights.size(), 1, 0);
	valueCheck(weights, OC, IC, KH, KW);

	// bias[OC]
	// 임시 bias 값
	std::vector<float> bias(OC);
	initTensor(bias, bias.size(), 1, 0);
	valueCheck(bias, OC, 1, 1, 1);

	// d[IN][IC][IH][IW] 
	// 임시 input 값 
	std::vector<float> inputs(N * IC * IH * IW);
	initTensor(inputs, inputs.size(), 1, 1);
	valueCheck(inputs, N, IC, IH, IW);

	// d[IN][OC][OH][OW] 
	// 임시 input 값 
	int OH = (IH - KH + TP + BP) / SH + 1; // output height
	int OW = (IW - KW + LP + RP) / SW + 1; // output width
	std::vector<float> outputs(N * OC * OH * OW);

	//[inputs]
	auto conv_src_md = memory::desc({ N, IC, IH, IW }, dt::f32, tag::nchw);
	auto user_src_memory = memory(conv_src_md, engine);
	write_to_dnnl_memory(inputs.data(), user_src_memory);

	memory output_s;
	conv2d_onednn_wo_bias2(output_s, user_src_memory,
		weights,
		net,
		net_args,
		engine, stream,
		N, IC, IH, IW, OC,
		KH, KW, SH, SW,
		TP, BP, LP, RP, 0);

	//memory output_s2;
	//conv2d_onednn(output_s2, output_s,
	//	weights, bias,
	//	net,
	//	net_args,
	//	engine, stream,
	//	N, IC, IH, IW, OC,
	//	KH, KW, SH, SW,
	//	TP, BP, LP, RP, 0);

	//[Execute model]
	for (int j = 0; j < 1; ++j) {
		assert(net.size() == net_args.size() && "something is missing");
		for (size_t i = 0; i < net.size(); ++i)
			net.at(i).execute(stream, net_args.at(i)); 
	}

	stream.wait();
	read_from_dnnl_memory(outputs.data(), net_args.at(net.size() - 1).find(DNNL_ARG_DST)->second);
	valueCheck(outputs, N, OC, OH, OW);
	std::cout << "done!!!" << std::endl;
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
