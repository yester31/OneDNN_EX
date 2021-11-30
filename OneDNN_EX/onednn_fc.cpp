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


void bn_onednn(
	memory &OUTPUT, memory &INPUT,
	std::vector<float> &scale_shift,
	std::vector<float> &mean,
	std::vector<float> &var,
	std::vector<primitive> &net,
	std::vector<std::unordered_map<int, memory>> &net_args,
	engine &engine, stream &engine_stream,
	int N, int IC,
	int Acti = 0)
{
	OUTPUT = memory(INPUT.get_desc(), engine);

	auto scale_shift_mem_md = memory::desc({ 2, IC }, dt::f32, tag::nc);
	auto scale_shift_mem = memory(scale_shift_mem_md, engine);
	write_to_dnnl_memory(scale_shift.data(), scale_shift_mem);

	auto mean_mem_md = memory::desc({ 1, IC }, dt::f32, tag::nc);
	auto mean_mem = memory(mean_mem_md, engine);
	write_to_dnnl_memory(mean.data(), mean_mem);

	auto variance_mem_md = memory::desc({ 1, IC }, dt::f32, tag::nc);
	auto variance_mem = memory(variance_mem_md, engine);
	write_to_dnnl_memory(var.data(), variance_mem);

	// Create primitive descriptor.
	batch_normalization_forward::primitive_desc bnorm_pd;

	if (Acti == 1) { // relu
		auto bnorm_d = batch_normalization_forward::desc(prop_kind::forward_inference, INPUT.get_desc(), 1.e-5f,
			normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu);
		bnorm_pd = batch_normalization_forward::primitive_desc(bnorm_d, engine);
	}
	else { // linear
		auto bnorm_d = batch_normalization_forward::desc(prop_kind::forward_inference, INPUT.get_desc(), 1.e-5f,
			normalization_flags::use_global_stats | normalization_flags::use_scale_shift);
		bnorm_pd = batch_normalization_forward::primitive_desc(bnorm_d, engine);
	}

	// Create the primitive.
	auto bnorm_prim = batch_normalization_forward(bnorm_pd);

	net.push_back(bnorm_prim);
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_MEAN, mean_mem },
		{ DNNL_ARG_VARIANCE, variance_mem },
		{ DNNL_ARG_SCALE_SHIFT, scale_shift_mem },
		{ DNNL_ARG_DST, OUTPUT }
		});
}

void pooling_onednn(
	memory &OUTPUT, memory &INPUT,
	std::vector<primitive> &net,
	std::vector<std::unordered_map<int, memory>> &net_args,
	engine &engine, stream &engine_stream,
	int N, int IC, int IH, int IW,
	int KH, int KW, int	SH, int SW,
	int TP, int BP, int LP, int RP,
	int DH, int DW, int mode,
	int Acti = 0)
{
	const memory::dim OH = (IH + TP + BP - (KH - 1) * DH - 1) / SH + 1;
	const memory::dim OW = (IW + LP + RP - (KW - 1) * DW - 1) / SW + 1;

	auto pooling_dst_md = memory::desc({ N, IC, OH, OW }, dt::f32, tag::nchw);
	OUTPUT = memory(pooling_dst_md, engine);

	pooling_v2_forward::primitive_desc pooling_pd;

	if (mode == 1) {//pooling_max
		auto pooling_d = pooling_v2_forward::desc(prop_kind::forward_inference, algorithm::pooling_max,
			INPUT.get_desc(), pooling_dst_md, { SH, SW }, { KH, KW }, { TP, LP }, { BP, RP }, { DH, DW });
		pooling_pd = pooling_v2_forward::primitive_desc(pooling_d, engine);
	}
	else {//pooling_avg
		auto pooling_d = pooling_v2_forward::desc(prop_kind::forward_inference, algorithm::pooling_avg,
			INPUT.get_desc(), pooling_dst_md, { SH, SW }, { KH, KW }, { DH, DW }, { TP, LP }, { BP, RP });
		pooling_pd = pooling_v2_forward::primitive_desc(pooling_d, engine);
	}

	auto workspace_mem = memory(pooling_pd.workspace_desc(), engine);

	// Create the primitive.
	auto pooling_prim = pooling_v2_forward(pooling_pd);

	net.push_back(pooling_prim);
	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_WORKSPACE, workspace_mem },
		{ DNNL_ARG_DST, OUTPUT }
		});

}

void gap_onednn(
	memory &OUTPUT, memory &INPUT,
	std::vector<primitive> &net,
	std::vector<std::unordered_map<int, memory>> &net_args,
	engine &engine, stream &engine_stream,
	int N, int IC,
	int Acti = 0)
{
	auto gap_dst_md = memory::desc({ N, IC ,1,1}, dt::f32, tag::nchw);
	OUTPUT = memory(gap_dst_md, engine);

	auto gap_d = reduction::desc(algorithm::reduction_mean, INPUT.get_desc(), gap_dst_md, 0.f, 0.f);
	reduction::primitive_desc gap_pd = reduction::primitive_desc(gap_d, engine);

	// Create the primitive.
	auto gap_prim = reduction(gap_pd);

	net.push_back(gap_prim);
	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_DST, OUTPUT }
		});
}

void fc_onednn(
	memory &OUTPUT, memory &INPUT,
	std::vector<float> &weights,
	std::vector<float> &bias,
	std::vector<primitive> &net,
	std::vector<std::unordered_map<int, memory>> &net_args,
	engine &engine, stream &engine_stream,
	int N, int IC, int OC,
	int Acti = 0)
{
	auto fc_dst_md = memory::desc({ N, OC }, dt::f32, tag::nc);
	OUTPUT = memory(fc_dst_md, engine);

	auto fc_weights_md = memory::desc({OC, IC,1,1}, dt::f32, tag::oihw);
	auto fc_weights_mem = memory(fc_weights_md, engine);
	write_to_dnnl_memory(weights.data(), fc_weights_mem);

	auto fc_bias_md = memory::desc({ OC }, dt::f32, tag::a);
	auto fc_bias_mem = memory(fc_bias_md, engine);
	write_to_dnnl_memory(bias.data(), fc_bias_mem);

	// Create operation descriptor.
	auto fc_desc = inner_product_forward::desc(prop_kind::forward_inference,
		INPUT.get_desc(), fc_weights_md, fc_bias_md, OUTPUT.get_desc());

	// Activation func
	inner_product_forward::primitive_desc fc_pd;

	if (Acti == 1) {
		// Create primitive post-ops (ReLU).
		const float scale = 1.f;
		const float alpha = 0.f;
		const float beta = 0.f;
		post_ops fc_ops;
		fc_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
		primitive_attr fc_attr;
		fc_attr.set_post_ops(fc_ops);
		fc_pd = inner_product_forward::primitive_desc(fc_desc, fc_attr, engine);
	}
	else { // linear
		fc_pd = inner_product_forward::primitive_desc(fc_desc, engine);
	}

	// Create the primitive.
	auto fc_prim = inner_product_forward(fc_pd);
	net.push_back(fc_prim);

	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_WEIGHTS, fc_weights_mem },
		{ DNNL_ARG_BIAS, fc_bias_mem },
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

	int OC = 3;
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
	//valueCheck(weights, OC, IC, KH, KW);

	// bias[OC]
	// 임시 bias 값
	std::vector<float> scale_shift(OC * 2);
	initTensor(scale_shift, scale_shift.size(), 2, 0);
	//valueCheck(scale_shift, OC, 1, 1, 1);

	std::vector<float> mean(OC);
	initTensor(mean, mean.size(), 2, 0);
	//valueCheck(mean, OC, 1, 1, 1);

	std::vector<float> var(OC);
	initTensor(var, var.size(), 2, 0);
	//valueCheck(var, OC, 1, 1, 1);

	// d[IN][IC][IH][IW] 
	// 임시 input 값 
	std::vector<float> inputs(N * IC * IH * IW);
	initTensor(inputs, inputs.size(), -5, 1);
	valueCheck(inputs, N, IC, IH, IW);

	// d[IN][OC][OH][OW] 
	// 임시 input 값 
	int OH = (IH - KH + TP + BP) / SH + 1; // output height
	int OW = (IW - KW + LP + RP) / SW + 1; // output width

	//[inputs]
	auto conv_src_md = memory::desc({ N, IC, IH, IW }, dt::f32, tag::nchw);
	auto user_src_memory = memory(conv_src_md, engine);
	write_to_dnnl_memory(inputs.data(), user_src_memory);

	memory output_s;
	conv2d_onednn_wo_bias(output_s, user_src_memory,
		weights,
		net, net_args, engine, stream,
		N, IC, IH, IW, OC,
		KH, KW, SH, SW,
		TP, BP, LP, RP, 0);

	memory output_s2;
	bn_onednn(output_s2, output_s,
		scale_shift, mean, var,
		net, net_args, engine, stream,
		N, OC, 0);

	memory output_s3;
	pooling_onednn(output_s3, output_s2, net, net_args, engine, stream,
		N, OC, OH, OW,
		2, 2, 2, 2,
		0, 0, 0, 0,
		1, 1, 1);

	memory output_s4;
	gap_onednn(output_s4, output_s3, net, net_args, engine, stream,
		N, OC);

	std::vector<float> weights2(OC * 10);
	initTensor(weights2, weights2.size(), 1, 0);
	//valueCheck(weights2, OC, 10, 1, 1);

	std::vector<float> bias2(10);
	initTensor(bias2, bias2.size(), 1, 0);
	//valueCheck(bias2, 10, 1, 1, 1);

	memory output_s5;
	fc_onednn(output_s5, output_s4, 
		weights2, bias2,
		net, net_args, engine, stream,
		N, OC, 10);
	
	std::vector<float> outputs(N * 10);

	//[Execute model]
	for (int j = 0; j < 1; ++j) {
		assert(net.size() == net_args.size() && "something is missing");
		for (size_t i = 0; i < net.size(); ++i)
			net.at(i).execute(stream, net_args.at(i));
	}

	stream.wait();
	read_from_dnnl_memory(outputs.data(), net_args.at(net.size() - 1).find(DNNL_ARG_DST)->second);
	valueCheck(outputs, N, 1, 1, 10);
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

