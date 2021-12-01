#include <assert.h>
#include <chrono>
#include <vector>
#include <unordered_map>
#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"
#include <iomanip>
#include "utils.h"
#include "weight_loader.h"
#include "data_loader.h"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

struct tensor_dims {int N;int IC;int IH;int IW;};

memory conv2d_onednn_wo_bias(
	memory &INPUT,  std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights,
	tensor_dims &t_dims, int OC, int KH, int KW, int SH, int SW, int TP, int BP, int LP, int RP, int Acti = 0);

memory bn_onednn(
	memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream, 
	std::vector<float> &scale, std::vector<float> &shift, std::vector<float> &mean, std::vector<float> &var,
	tensor_dims &t_dims, float eps = 1.e-5f, int Acti = 0);

memory pooling_onednn(
	memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	tensor_dims &t_dims, int KH, int KW, int SH, int SW, int DH, int DW, int TP, int BP, int LP, int RP, int mode, int Acti = 0);

memory gap_onednn(
	memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	tensor_dims &t_dims, int Acti = 0);

memory fc_onednn(
	memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights, std::vector<float> &bias, 
	tensor_dims &t_dims, int OC, int Acti = 0);



void resnet18(engine::kind engine_kind, int times = 100) {

	// Weight load =============================================================
	std::string file = "../model/resnet18.wts";
	std::map<std::string, Weights> weightMap = loadWeights(file);
	std::cout << "weight load done!" << std::endl;

	// Image load =============================================================
	std::string img_dir = "../data";
	std::vector<std::string> file_names;
	if (SearchFile(img_dir.c_str(), file_names) < 0) {
		std::cerr << "data search error" << std::endl;
		exit(0);
	}

	// Image preprocess ============================================================
	int batch_size = 1;
	int input_channel = 3;
	int input_width = 224;
	int input_height = 224;
	cv::Mat img(input_height, input_width, CV_8UC3);
	cv::Mat ori_img;
	std::vector<uint8_t> input(batch_size * input_height * input_width * input_channel);
	std::vector<float> inputs(input.size());
	for (int idx = 0; idx < batch_size; idx++) {
		cv::Mat ori_img = cv::imread(file_names[idx]);
		cv::resize(ori_img, img, img.size(), cv::INTER_LINEAR);
		int offset = idx * input_height * input_width * input_channel;
		memcpy(input.data() + offset, img.data, input_height * input_width * input_channel);
	}

	preprocess(inputs, input, batch_size, input_channel, input_height, input_width);
	//tofile(inputs);

	// ONEDNN =============================================================
	//[Initialize engine and stream]
	engine engine(engine_kind, 0);
	stream stream(engine);

	//[Create network]
	std::vector<primitive> net;
	std::vector<std::unordered_map<int, memory>> net_args;

	//[inputs]
	tensor_dims t_dims{ batch_size , input_channel, input_height, input_width };
	auto inputs_src_md = memory::desc({ batch_size, input_channel, input_height, input_width}, dt::f32, tag::nchw);
	auto inputs_src_md_memory = memory(inputs_src_md, engine);
	write_to_dnnl_memory(inputs.data(), inputs_src_md_memory);

	// net work
	memory conv1 = conv2d_onednn_wo_bias(inputs_src_md_memory, net, net_args, engine, stream, weightMap["conv1.weight"].values,  t_dims, 64, 7, 7, 2, 2, 3, 3, 3, 3, 0);
	memory bn1_relu1 = bn_onednn(conv1, net, net_args, engine, stream, weightMap["bn1.weight"].values, weightMap["bn1.bias"].values, weightMap["bn1.running_mean"].values, weightMap["bn1.running_var"].values, t_dims, 1.e-5f, 1);
	
	// 검증 필요 Padding 문제 발견
	memory pool1 = pooling_onednn(bn1_relu1, net, net_args, engine, stream, t_dims, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0); 

	//[Execute model]
	for (int j = 0; j < 1; ++j) {
		assert(net.size() == net_args.size() && "something is missing");
		for (size_t i = 0; i < net.size(); ++i)
			net.at(i).execute(stream, net_args.at(i));
	}
	stream.wait();

	int output_size = t_dims.N * t_dims.IC* t_dims.IH* t_dims.IW;
	std::vector<float> outputs(output_size);
	read_from_dnnl_memory(outputs.data(), net_args.at(net.size() - 1).find(DNNL_ARG_DST)->second);
	tofile(outputs);
	//valueCheck(outputs, batch_size, 64, 56*2, 56*2);
	std::cout << "done!!!" << std::endl;
	std::cout << "layer count : " << net.size() << std::endl;
}

void cnn_inference_f32(engine::kind engine_kind) {

	auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	int times = 100;
	resnet18(engine_kind, times);
	auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	std::cout << "Use time: " << (end - begin) << " ms per iteration." << std::endl;
}

int main(int argc, char **argv) {
	return handle_example_errors(cnn_inference_f32, parse_engine_kind(argc, argv));
}


memory conv2d_onednn_wo_bias(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream, 
	std::vector<float> &weights, tensor_dims &t_dims, int OC, int KH, int KW, int SH, int SW, int TP, int BP, int LP, int RP, int Acti)
{
	int OH = (t_dims.IH - KH + TP + BP) / SH + 1; // output height
	int OW = (t_dims.IW - KW + LP + RP) / SW + 1; // output width

	auto conv_dst_md = memory::desc({ t_dims.N, OC, OH, OW }, dt::f32, tag::nchw);
	memory OUTPUT = memory(conv_dst_md, engine);

	auto conv_weights_md = memory::desc({ OC, t_dims.IC, KH, KW }, dt::f32, tag::oihw);
	auto user_weights_mem = memory(conv_weights_md, engine);
	write_to_dnnl_memory(weights.data(), user_weights_mem);

	// Create operation descriptor.
	auto conv_desc = convolution_forward::desc(prop_kind::forward_inference, algorithm::convolution_auto, INPUT.get_desc(), conv_weights_md, conv_dst_md, { SH, SW }, { TP, LP }, { BP, RP });

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

	t_dims.IC = OC;
	t_dims.IH = OH;
	t_dims.IW = OW;

	return OUTPUT;
}

memory bn_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream, 
	std::vector<float> &scale, std::vector<float> &shift, std::vector<float> &mean, std::vector<float> &var, tensor_dims &t_dims, float eps, int Acti)
{
	memory OUTPUT = memory(INPUT.get_desc(), engine);

	std::vector<float> scale_shift(2 * t_dims.IC);
	memcpy(scale_shift.data(), scale.data(), sizeof(float) * t_dims.IC);
	memcpy(scale_shift.data()+ t_dims.IC, shift.data(), sizeof(float) * t_dims.IC);

	auto scale_shift_mem_md = memory::desc({ 2, t_dims.IC }, dt::f32, tag::nc);
	auto scale_shift_mem = memory(scale_shift_mem_md, engine);
	write_to_dnnl_memory(scale_shift.data(), scale_shift_mem);

	auto mean_mem_md = memory::desc({ 1, t_dims.IC }, dt::f32, tag::nc);
	auto mean_mem = memory(mean_mem_md, engine);
	write_to_dnnl_memory(mean.data(), mean_mem);

	auto variance_mem_md = memory::desc({ 1, t_dims.IC }, dt::f32, tag::nc);
	auto variance_mem = memory(variance_mem_md, engine);
	write_to_dnnl_memory(var.data(), variance_mem);

	// Create primitive descriptor.
	batch_normalization_forward::primitive_desc bnorm_pd;

	if (Acti == 1) { // relu
		auto bnorm_d = batch_normalization_forward::desc(prop_kind::forward_inference, INPUT.get_desc(), eps,
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
	return OUTPUT;
}

memory pooling_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	tensor_dims &t_dims, int KH, int KW, int SH, int SW, int DH, int DW, int TP, int BP, int LP, int RP, int mode, int Acti)
{
	//const memory::dim OH = (t_dims.IH + (TP + BP) - DH * (KH - 1) - 1) / SH + 1;
	//const memory::dim OW = (t_dims.IW + (LP + RP) - DW * (KW - 1) - 1) / SW + 1;

	const memory::dim OH = (t_dims.IH + (TP + BP) - (DH * (KH - 1) + KH)) / SH + 1;
	const memory::dim OW = (t_dims.IW + (LP + RP) - (DW * (KW - 1) + KW)) / SW + 1;

	auto pooling_dst_md = memory::desc({ t_dims.N, t_dims.IC, OH, OW }, dt::f32, tag::nchw);
	memory OUTPUT = memory(pooling_dst_md, engine);

	pooling_v2_forward::primitive_desc pooling_pd;

	if (mode == 1) {//pooling_max
		auto pooling_d = pooling_v2_forward::desc(prop_kind::forward_inference, algorithm::pooling_max,
			INPUT.get_desc(), pooling_dst_md, { SH, SW }, { KH, KW }, { DH, DW }, { TP, LP }, { BP, RP });
		pooling_pd = pooling_v2_forward::primitive_desc(pooling_d, engine);
	}
	else {//pooling_avg
		auto pooling_d = pooling_v2_forward::desc(prop_kind::forward_inference, algorithm::pooling_avg,
			INPUT.get_desc(), pooling_dst_md, { SH, SW }, { KH, KW }, { DH, DW }, { TP, LP }, { BP, RP });
		pooling_pd = pooling_v2_forward::primitive_desc(pooling_d, engine);
	}

	// Create the primitive.
	auto pooling_prim = pooling_v2_forward(pooling_pd);

	net.push_back(pooling_prim);
	// Primitive arguments.
	net_args.push_back({
		{ DNNL_ARG_SRC, INPUT },
		{ DNNL_ARG_DST, OUTPUT }
		});

	t_dims.IH = OH;
	t_dims.IW = OW;

	return OUTPUT;
}

memory gap_onednn(memory &INPUT, std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	tensor_dims &t_dims, int Acti)
{
	auto gap_dst_md = memory::desc({ t_dims.N, t_dims.IC ,1,1 }, dt::f32, tag::nchw);
	memory OUTPUT = memory(gap_dst_md, engine);

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
	t_dims.IH = 1;
	t_dims.IW = 1;
	return OUTPUT;
}

memory fc_onednn(memory &INPUT,  std::vector<primitive> &net, std::vector<std::unordered_map<int, memory>> &net_args, engine &engine, stream &engine_stream,
	std::vector<float> &weights, std::vector<float> &bias, tensor_dims &t_dims, int OC, int Acti)
{
	auto fc_dst_md = memory::desc({ t_dims.N, OC }, dt::f32, tag::nc);
	memory OUTPUT = memory(fc_dst_md, engine);

	auto fc_weights_md = memory::desc({ OC, t_dims.IC,1,1 }, dt::f32, tag::oihw);
	auto fc_weights_mem = memory(fc_weights_md, engine);
	write_to_dnnl_memory(weights.data(), fc_weights_mem);

	auto fc_bias_md = memory::desc({ OC }, dt::f32, tag::a);
	auto fc_bias_mem = memory(fc_bias_md, engine);
	write_to_dnnl_memory(bias.data(), fc_bias_mem);

	// Create operation descriptor.
	auto fc_desc = inner_product_forward::desc(prop_kind::forward_inference, INPUT.get_desc(), fc_weights_md, fc_bias_md, OUTPUT.get_desc());

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

	t_dims.IC = OC;
	return OUTPUT;
}
