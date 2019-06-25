#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "tensor.h"
#define TORCH_MODULE_(Name,T) template<typename T>\
	  class Name : public torch::nn::ModuleHolder<Name##Impl<T>> { /* NOLINT */ \
   public:                                                         \
    using torch::nn::ModuleHolder<Name##Impl<T>>::ModuleHolder;             \
  }
#define TORCH_MODULE_E(Name,T,E) template<typename T>\
	  class Name : public torch::nn::ModuleHolder<Name##Impl<T>> { /* NOLINT */ \
    public:                                                         \
	static const int64_t expansion = E;                     \
    using torch::nn::ModuleHolder<Name##Impl<T>>::ModuleHolder;             \
  }
#define TORCH_MODULE__(Name,T1,T2) template<typename T1,typename T2>\
	  class Name : public torch::nn::ModuleHolder<Name##Impl<T1,T2>> { /* NOLINT */ \
   public:                                                         \
    using torch::nn::ModuleHolder<Name##Impl<T1,T2>>::ModuleHolder;             \
  }
void PrintfTensor(torch::Tensor& t) {
	auto size = t.sizes().vec();
	std::string str = "";
	for (auto s : size) {
		printf("%d,", s);
	}
	printf(t.dtype().name());
	printf(",");
	printf("%d", t.numel());
	printf(",");
}
void PrintTensorUInt8_tValue(torch::Tensor& t) {
	auto size = t.sizes().vec();
	std::string str = "Start:";
	for (auto s : size) {
		printf("%d,", s);
	}
	printf(t.dtype().name());
	printf(",");
	printf("%d", t.numel());
	printf(",Value:");
	int64_t l = t.numel();
	uint8_t* values = new uint8_t[l];
	auto s = t.cpu();
	memcpy(values, s.data_ptr(), s.nbytes());
	for (int64_t i = 0; i < l; i++) {
		printf("%d,", values[i]);
	}
	printf(",ValueEnd");
	delete[] values;
}
void PrintTensorFloatValue(torch::Tensor& t) {
	auto size = t.sizes().vec();
	std::string str = "Start:";
	for (auto s : size) {
		printf("%d,", s);
	}
	printf(t.dtype().name());
	printf(",");
	printf("%d", t.numel());
	printf(",Value:");
	int64_t l = t.numel();
	float* values = new float[l];
	auto s = t.cpu();
	memcpy(values, s.data_ptr(), s.nbytes());
	for (int64_t i = 0; i < l; i++) {
		printf("%f,", values[i]);
	}
	printf(",ValueEnd");
	delete[] values;
}
void PrintTensorInt64_tValue(torch::Tensor& t) {
	auto size = t.sizes().vec();
	std::string str = "Start:";
	for (auto s : size) {
		printf("%d,", s);
	}
	printf(t.dtype().name());
	printf(",");
	printf("%d", t.numel());
	printf(",Value:");
	int64_t l = t.numel();
	int64_t* values = new int64_t[l];
	auto s = t.cpu();
	memcpy(values, s.data_ptr(), s.nbytes());
	for (int64_t i = 0; i < l; i++) {
		printf("%lld,", values[i]);
	}
	printf(",ValueEnd");
	delete[] values;
}
void fix_model(torch::nn::Module m, std::set<std::string> train_param_name) {
	auto ps = m.named_parameters();
	for (auto p : ps) {
		p.value().set_requires_grad(train_param_name.find(p.key()) != train_param_name.end());
	}
}
template <typename Module>
std::shared_ptr<Module> AllNet(bool pretrained, int64_t num_classes, const std::string& file_name) {
	auto model = std::make_shared<Module>(num_classes);
	if (pretrained) {
		printf("loading pretrained model.....");
		try {
			torch::load(model, file_name);
		}
		catch (c10::Error e) {
			printf(e.msg().c_str());
		}
	}
	return model;
}
template <typename Module>
Module UsedNet(bool pretrained, int64_t num_classes, const std::string& file_name) {
	auto model = Module(num_classes);
	if (pretrained) {
		printf("loading pretrained model.....");
		try {
			torch::load(model, file_name);
		}
		catch (c10::Error e) {
			printf(e.msg().c_str());
		}
	}
	return model;
}
torch::nn::Conv2d Conv2d(int64_t inplanes, int64_t planes, int64_t kernal_size,
	int64_t stride = 1, int64_t pading = 0, int64_t groups = 1, int64_t dilation = 1, bool bias = true) {
	auto co = torch::nn::Conv2dOptions(inplanes, planes, kernal_size);
	co.stride(stride);
	co.padding(pading);
	co.with_bias(bias);
	co.groups(groups);
	co.dilation(dilation);
	return torch::nn::Conv2d(co);
}
torch::Tensor fixed_padding(const torch::Tensor& inputs, int64_t kernel_size, int64_t dilation) {
	int64_t kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1);
	int64_t pad_total = kernel_size_effective - 1;
	int64_t pad_beg = pad_total;
	int64_t pad_end = pad_total - pad_beg;
	std::vector<int64_t> pads;
	pads.push_back(pad_beg);
	pads.push_back(pad_end);
	pads.push_back(pad_beg);
	pads.push_back(pad_end);
	torch::Tensor padded_inputs = torch::constant_pad_nd(inputs, pads);
	return padded_inputs;
}

struct SeparableConv2dImpl : torch::nn::Cloneable<SeparableConv2dImpl> {
	SeparableConv2dImpl(int64_t inplanes, int64_t planes, int64_t kernel_size = 3,
		int64_t stride = 1, int64_t dilation = 1, bool bias = false) {
		ks = kernel_size;
		dl = dilation;
		conv = Conv2d(inplanes, inplanes, kernel_size, stride, 0, inplanes, dilation, bias);
		bn = torch::nn::BatchNorm(inplanes);
		pointwise = Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias);
		register_module("conv1", conv);
		register_module("bn1", bn);
		register_module("pointwise1", pointwise);
	}

	torch::Tensor forward(const torch::Tensor& x) {
		auto out = fixed_padding(x, ks, dl);
		out = conv(out);
		out = bn(out);
		out = pointwise(out);
		return out;
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override{
		stream << "::SeparableConv2d";
	}
	int64_t ks;
	int64_t dl;
	torch::nn::Conv2d conv = nullptr;
	torch::nn::BatchNorm bn = nullptr;
	torch::nn::Conv2d pointwise = nullptr;
};
TORCH_MODULE(SeparableConv2d);
struct BlockImpl:torch::nn::Cloneable<BlockImpl> {
	BlockImpl(int64_t inplanes, int64_t planes, int64_t stride = 1,
		int64_t dilation = 1, bool start_with_relu = true) {
		if (planes != inplanes || stride != 1) {
			skip = Conv2d(inplanes, planes, 1, stride, 0, 1, 1, false);
			skipbn = torch::nn::BatchNorm(planes);
			register_module("skip", skip);
			register_module("skipbn", skipbn);
		}
		int64_t inter_channels = std::floor(planes / 4);
		rep = torch::nn::Sequential();
		if (start_with_relu)
			rep->push_back(torch::nn::Functional(torch::relu));
		rep->push_back(SeparableConv2d(inplanes, inter_channels, 3, 1, dilation));
		rep->push_back(torch::nn::BatchNorm(inter_channels));
		rep->push_back(torch::nn::Functional(torch::relu));

		rep->push_back(SeparableConv2d(inter_channels, inter_channels, 3, 1, dilation));
		rep->push_back(torch::nn::BatchNorm(inter_channels));
		if (stride != 1) {
			rep->push_back(torch::nn::Functional(torch::relu));
			rep->push_back(SeparableConv2d(inter_channels, planes, 3, 2));
			rep->push_back(torch::nn::BatchNorm(planes));
		}
		else {
			rep->push_back(torch::nn::Functional(torch::relu));
			rep->push_back(SeparableConv2d(inter_channels, planes, 3, 1));
			rep->push_back(torch::nn::BatchNorm(planes));
		}
		register_module("rep", rep);
	}

	torch::Tensor forward(const torch::Tensor& x) {
		auto out = rep->forward(x);
		torch::Tensor s=x;
		if (!skip.is_empty()) 
			s = skipbn(skip(x));
		out = out + s;
		return out;
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::Block";
	}
	torch::nn::Conv2d skip = nullptr;
	torch::nn::BatchNorm skipbn = nullptr;
	torch::nn::Sequential rep = nullptr;
};
TORCH_MODULE(Block);
struct EncImpl : torch::nn::Cloneable<EncImpl> {
	EncImpl(int64_t inplanes, int64_t planes, int64_t stage) {
		reps = torch::nn::Sequential(
			Block(inplanes, planes, 2, 1,false)
		);
		for (int64_t i = 0; i < stage - 1; i++)
			reps->push_back(Block(planes, planes, 1));
		register_module("reps1", reps);
	}
	torch::Tensor forward(const torch::Tensor& x) {
		return reps->forward(x);
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::Enc";
	}
	torch::nn::Sequential reps=nullptr;
};
TORCH_MODULE(Enc);
struct FCAttentionImpl : torch::nn::Cloneable<FCAttentionImpl> {
	FCAttentionImpl(int64_t inplanes) {
		std::vector<int64_t> a = { 1,1 };
		avg_pool = torch::nn::Functional(torch::adaptive_avg_pool2d, a);
		fc = torch::nn::Sequential(
			torch::nn::Linear(torch::nn::LinearOptions(inplanes, 1000).with_bias(false))
		);
		conv = torch::nn::Sequential(Conv2d(1000, inplanes, 1, 1, 0, 1, 1, false),
			torch::nn::BatchNorm(inplanes),
			torch::nn::Functional(torch::relu)
		);
		register_module("avg_pool1", avg_pool);
		register_module("fc1", fc);
		register_module("conv1", conv);
	}

	torch::Tensor forward(const torch::Tensor& x) {
		auto s = x.sizes();
		auto ap = avg_pool(x);
		auto att = ap.view({ s[0],s[1] });
		att = fc->forward(att).view({ s[0],1000,1,1 });
		att = conv->forward(att);
		return x*att.expand_as(x);
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::FCAttention";
	}
	torch::nn::Functional avg_pool = nullptr;
	torch::nn::Sequential fc = nullptr;
	torch::nn::Sequential conv = nullptr;
};
TORCH_MODULE(FCAttention);
struct _ConvBNReLUImpl :torch::nn::Cloneable<_ConvBNReLUImpl> {
	_ConvBNReLUImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
		int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1, int64_t groups = 1,
		bool relu6 = false) {
		conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups, dilation, false);
		bn = torch::nn::BatchNorm(out_channels);
		if (relu6) 
			relu = torch::nn::Functional(torch::hardtanh_, -1.0f,1.0f);
		else
			relu = torch::nn::Functional(torch::relu);
		register_module("conv", conv);
		register_module("bn", bn);
		register_module("relu", relu);
	}
	torch::Tensor forward(const torch::Tensor& input) {
		auto x = conv->forward(input);
		x = bn->forward(x);
		x = relu->forward(x);
		return x;
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::_ConvBNReLU";
	}
	torch::nn::Conv2d conv = nullptr;
	torch::nn::BatchNorm bn = nullptr;
	torch::nn::Functional relu = nullptr;
};
TORCH_MODULE(_ConvBNReLU);
struct XceptionAImpl : torch::nn::Cloneable<XceptionAImpl> {
	XceptionAImpl(int64_t num_classes) {
		conv1 = _ConvBNReLU(3, 8, 3, 2, 1);
		enc2 = Enc(8, 48, 4);
		enc3 = Enc(48, 96, 6);
		enc4 = Enc(96, 192, 4);
		fca = FCAttention(192);
		std::vector<int64_t> a = { 1,1 };
		avg_pool = torch::nn::Functional(torch::adaptive_avg_pool2d, a);
		fc = torch::nn::Linear(192, num_classes);
		register_module("conv1", conv1);
		register_module("enc2", enc2);
		register_module("enc3", enc3);
		register_module("enc4", enc4);
		register_module("fca", fca);
		register_module("avg_pool", avg_pool);
		register_module("fc", fc);
	}

	torch::Tensor forward(const torch::Tensor& input) {
		auto s = input.sizes();
		auto x = conv1->forward(input);
		x = enc2->forward(x);
		x = enc3->forward(x);
		x = enc4->forward(x);
		x = fca->forward(x);
		x = avg_pool->forward(x).view({ s[0],-1 });
		x = fc->forward(x);
		return x;
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::XceptionA";
	}
	_ConvBNReLU conv1 = nullptr;
	Enc enc2 = nullptr;
	Enc enc3 = nullptr;
	Enc enc4 = nullptr;
	FCAttention fca = nullptr;
	torch::nn::Functional avg_pool = nullptr;
	torch::nn::Linear fc = nullptr;
};
TORCH_MODULE(XceptionA);
struct XceptionAx3Impl : torch::nn::Cloneable<XceptionAx3Impl> {
	XceptionAx3Impl(int64_t num_classes) {
		conv1 = _ConvBNReLU(3, 8, 3, 2, 1);
		enc2a = Enc(8, 48, 4);
		enc2b = Enc(240, 48, 4);
		enc2c = Enc(240, 48, 4);
		enc3a = Enc(48, 96, 6);
		enc3b = Enc(144, 96, 6);
		enc3c = Enc(144, 96, 6);
		enc4a = Enc(96, 192, 4);
		enc4b = Enc(288, 192, 4);
		enc4c = Enc(288, 192, 4);
		fca1 = FCAttention(192);
		fca2 = FCAttention(192);
		fca3 = FCAttention(192);
		enc2a_to_decoder_dim_reduction = _ConvBNReLU(48, 32, 1);
		enc2b_to_decoder_dim_reduction = _ConvBNReLU(48, 32, 1);
		enc2c_to_decoder_dim_reduction = _ConvBNReLU(48, 32, 1);
		merge_conv = _ConvBNReLU(32, 32, 1);
		fca1_to_decoder_dim_reduction = _ConvBNReLU(192, 32, 1);
		fca2_to_decoder_dim_reduction = _ConvBNReLU(192, 32, 1);
		fca3_to_decoder_dim_reduction = _ConvBNReLU(192, 32, 1);
		last_conv = Conv2d(32, num_classes, 1, 1, 0, 1, 1, false);
		register_module("conv1", conv1);
		register_module("enc2a", enc2a);
		register_module("enc2b", enc2b);
		register_module("enc2c", enc2c);
		register_module("enc3a", enc3a);
		register_module("enc3b", enc3b);
		register_module("enc3c", enc3c);
		register_module("enc4a", enc4a);
		register_module("enc4b", enc4b);
		register_module("enc4c", enc4c);
		register_module("fca1", fca1);
		register_module("fca2", fca2);
		register_module("fca3", fca3);
		register_module("enc2a_to_decoder_dim_reduction", enc2a_to_decoder_dim_reduction);
		register_module("enc2b_to_decoder_dim_reduction", enc2b_to_decoder_dim_reduction);
		register_module("enc2c_to_decoder_dim_reduction", enc2c_to_decoder_dim_reduction);
		register_module("fca1_to_decoder_dim_reduction", fca1_to_decoder_dim_reduction);
		register_module("fca2_to_decoder_dim_reduction", fca2_to_decoder_dim_reduction);
		register_module("fca3_to_decoder_dim_reduction", fca3_to_decoder_dim_reduction);
		register_module("merge_conv", merge_conv);
		register_module("last_conv", last_conv);
	}
	torch::Tensor forward(const torch::Tensor& x) {
		//a
		auto stage1 = conv1->forward(x);
		auto stage_enc2a = enc2a->forward(stage1);
		auto stage_enc3a = enc3a->forward(stage_enc2a);
		auto stage_enc4a = enc4a->forward(stage_enc3a);
		auto stage_fca1 = fca1->forward(stage_enc4a);
		auto size = stage_enc2a.sizes().slice(2);
		auto up_fca1 = torch::upsample_bilinear2d(stage_fca1, size, false);
		//b
		auto stage_enc2b = enc2b->forward(torch::cat({ up_fca1 ,stage_enc2a },1));
		auto stage_enc3b = enc3b->forward(torch::cat({ stage_enc2b ,stage_enc3a }, 1));
		auto stage_enc4b = enc4b->forward(torch::cat({ stage_enc3b ,stage_enc4a }, 1));
		auto stage_fca2 = fca2->forward(stage_enc4b);
		auto size2 = stage_enc2b.sizes().slice(2);
		auto up_fca2 = torch::upsample_bilinear2d(stage_fca2, size2, false);

		//c
		auto stage_enc2c = enc2c->forward(torch::cat({ up_fca2 ,stage_enc2b }, 1));
		auto stage_enc3c = enc3c->forward(torch::cat({ stage_enc2c ,stage_enc3b }, 1));
		auto stage_enc4c = enc4c->forward(torch::cat({ stage_enc3c ,stage_enc4b }, 1));
		auto stage_fca3 = fca3->forward(stage_enc4c);
		
		//decoder
		auto x1 = enc2a_to_decoder_dim_reduction->forward(stage_enc2a);
		auto x2 = enc2b_to_decoder_dim_reduction->forward(stage_enc2b);
		auto x3 = enc2c_to_decoder_dim_reduction->forward(stage_enc2c);
		auto size3 = x1.sizes().slice(2);
		auto x2_up = torch::upsample_bilinear2d(x2, size3, false);	
		auto x3_up = torch::upsample_bilinear2d(x3, size3, false);
		auto x_up = x1 + x2_up + x3_up;
		auto x_merge = merge_conv->forward(x_up);
		auto x_fca1 = fca1_to_decoder_dim_reduction->forward(stage_fca1);
		auto x_fca2 = fca2_to_decoder_dim_reduction->forward(stage_fca2);
		auto x_fca3 = fca2_to_decoder_dim_reduction->forward(stage_fca3);
		auto x_fca1_up= torch::upsample_bilinear2d(x_fca1, size3, false);
		auto x_fca2_up = torch::upsample_bilinear2d(x_fca2, size3, false);
		auto x_fca3_up = torch::upsample_bilinear2d(x_fca3, size3, false);
		auto x_fca_up = x_merge + x_fca1_up + x_fca2_up + x_fca3_up;
		auto result = last_conv->forward(x_fca_up);
		auto xsize = x.sizes().slice(2);
		result= torch::upsample_bilinear2d(result, xsize, false);
		return result;
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::XceptionAx3";
	}
	_ConvBNReLU conv1 = nullptr;
	Enc enc2a = nullptr;
	Enc enc2b = nullptr;
	Enc enc2c = nullptr;
	Enc enc3a = nullptr;
	Enc enc3b = nullptr;
	Enc enc3c = nullptr;
	Enc enc4a = nullptr;
	Enc enc4b = nullptr;
	Enc enc4c = nullptr;
	FCAttention fca1 = nullptr;
	FCAttention fca2 = nullptr;
	FCAttention fca3 = nullptr;
	_ConvBNReLU enc2a_to_decoder_dim_reduction = nullptr;
	_ConvBNReLU enc2b_to_decoder_dim_reduction = nullptr;
	_ConvBNReLU enc2c_to_decoder_dim_reduction = nullptr;
	_ConvBNReLU fca1_to_decoder_dim_reduction = nullptr;
	_ConvBNReLU fca2_to_decoder_dim_reduction = nullptr;
	_ConvBNReLU fca3_to_decoder_dim_reduction = nullptr;
	_ConvBNReLU merge_conv = nullptr;
	torch::nn::Conv2d last_conv = nullptr;
};
TORCH_MODULE(XceptionAx3);
struct ASPPConvImpl :torch::nn::Cloneable<ASPPConvImpl> {
	ASPPConvImpl(int64_t in_channels, int64_t out_channels, int64_t dilation) {
		seq = torch::nn::Sequential(
			Conv2d(in_channels, out_channels, 3, 1, dilation, 1, dilation, false),
			torch::nn::BatchNorm(out_channels),
			torch::nn::Functional(torch::relu)
		);
	}
	torch::Tensor forward(const torch::Tensor& input) {
		return seq->forward(input);
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::ASPPConv";
	}
	torch::nn::Sequential seq = nullptr;
};
TORCH_MODULE(ASPPConv);
struct ASPPPoolingImpl :torch::nn::Cloneable<ASPPPoolingImpl> {
	ASPPPoolingImpl(int64_t in_channels, int64_t out_channels) {
		seq = torch::nn::Sequential();
		seq->push_back(Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, false));
		seq->push_back(torch::nn::BatchNorm(out_channels));
		seq->push_back(torch::nn::Functional(torch::relu));
		register_module("seq1", seq);
	}
	torch::Tensor forward(const torch::Tensor& input) {
		auto size = input.sizes();
		size = size.slice(size.size() - 2);
		auto x = torch::adaptive_avg_pool3d(input, { 1,1 });
		x = seq->forward(x);
		return torch::upsample_bilinear2d(x, size, false);
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::ASPPPooling";
	}
	torch::nn::Sequential seq = nullptr;
};
TORCH_MODULE(ASPPPooling);
struct ASPPListImpl :torch::nn::Cloneable<ASPPListImpl> {
	ASPPListImpl(int64_t in_channels, int64_t* atrous_rates) {
		int64_t out_channels = 256;
		seq = torch::nn::Sequential();
		seq->push_back(Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, false));
		seq->push_back(torch::nn::BatchNorm(out_channels));
		seq->push_back(torch::nn::Functional(torch::relu));
		conv1 = ASPPConv(in_channels, out_channels, atrous_rates[0]);
		conv2 = ASPPConv(in_channels, out_channels, atrous_rates[1]);
		conv3 = ASPPConv(in_channels, out_channels, atrous_rates[2]);
		pool = ASPPPooling(in_channels, out_channels);
		register_module("seq", seq);
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("pool", pool);
	}
	torch::Tensor forward(const torch::Tensor& x) {
		std::vector<torch::Tensor> ts;
		ts.push_back(seq->forward(x));
		ts.push_back(conv1->forward(x));
		ts.push_back(conv2->forward(x));
		ts.push_back(conv3->forward(x));
		ts.push_back(pool->forward(x));
		return torch::cat(ts, 1);
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::ASPPList";
	}
	torch::nn::Sequential seq = nullptr;
	ASPPConv conv1 = nullptr;
	ASPPConv conv2 = nullptr;
	ASPPConv conv3 = nullptr;
	ASPPPooling pool = nullptr;
};
TORCH_MODULE(ASPPList);
struct ASPPImpl :torch::nn::Cloneable<ASPPImpl> {
	ASPPImpl(int64_t in_channels, int64_t* atrous_rates) {
		int64_t out_channels = 256;
		al = ASPPList(in_channels, atrous_rates);
		project = torch::nn::Sequential();
		project->push_back(Conv2d(5 * out_channels, out_channels, 1, 1, 0, 1, 1, false));
		project->push_back(torch::nn::BatchNorm(out_channels));
		project->push_back(torch::nn::Functional(torch::relu));
		project->push_back(torch::nn::Dropout(0.5));
		register_module("al", al);
		register_module("project", project);
	}
	torch::Tensor forward(const torch::Tensor& x) {
		auto c = al->forward(x);
		return project->forward(c);
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::ASPP";
	}
	ASPPList al = nullptr;
	torch::nn::Sequential project = nullptr;
};
TORCH_MODULE(ASPP);
struct DeepLabHeadImpl :torch::nn::Cloneable<DeepLabHeadImpl> {
	DeepLabHeadImpl(int64_t in_channels, int64_t num_classes) {
		int64_t atrous_rates[4];
		atrous_rates[0] = 12;
		atrous_rates[1] = 24;
		atrous_rates[2] = 36;
		seq = torch::nn::Sequential(
			ASPP(in_channels, atrous_rates),
			Conv2d(256, 256, 3, 1, 1, 1, 1, false),
			torch::nn::BatchNorm(256),
			torch::nn::Functional(torch::relu),
			Conv2d(256, num_classes, 1, 1, 0, 1, 1, false)
		);
	}
	torch::Tensor forward(const torch::Tensor& input) {
		return seq->forward(input);
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::DeepLabHead";
	}
	torch::nn::Sequential seq = nullptr;
};
TORCH_MODULE(DeepLabHead);
torch::nn::Conv2d Conv3_3(int64_t inplanes, int64_t planes,
	int64_t stride = 1, int64_t groups = 1, int64_t dilation = 1) {
	return Conv2d(inplanes, planes, 3, stride, dilation, groups, dilation, false);
}
torch::nn::Conv2d Conv1_1(int64_t inplanes, int64_t planes,
	int64_t stride = 1) {
	return Conv2d(inplanes, planes, 1, stride, 0, 1, 1, false);
}
template <typename NormLayer >
struct BasicBlockImpl :torch::nn::Cloneable<BasicBlockImpl<NormLayer>> {
	BasicBlockImpl(int64_t inplanes, int64_t planes, int64_t instride = 1, torch::nn::Sequential ds = nullptr,
		int64_t groups = 1, int64_t base_width = 64, int64_t dilation = 1) {
		if (groups != 1)
			groups = 1;
		if (base_width != 64)
			base_width = 64;
		if (dilation > 1)
			dilation = 1;
		conv1 = Conv3_3(inplanes, planes, instride);
		bn1 = NormLayer(planes);
		relu = torch::nn::Functional(torch::relu);
		conv2 = Conv3_3(planes, planes);
		bn2 = NormLayer(planes);
		downsample = ds;
		stride = instride;
		register_module("conv1", conv1);
		register_module("bn1", bn1);
		register_module("conv2", conv2);
		register_module("bn2", bn2);
		register_module("relu", relu);
		if(!downsample.is_empty()) {
			register_module("downsample", downsample);
		}
	}
	torch::Tensor forward(const torch::Tensor& x) {
		auto identity = x;
		auto out = conv1(x);
		out = bn1(out);
		out = relu(out);
		out = conv2(out);
		out = bn2(out);
		if (!downsample.is_empty()) {
			identity = downsample->forward(x);
		}
		out += identity;
		out = relu(out);
		return out;
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::BasicBlock";
	}
	torch::nn::Conv2d conv1 = nullptr;
	torch::nn::BatchNorm bn1 = nullptr;
	torch::nn::Functional relu = nullptr;
	torch::nn::Conv2d conv2 = nullptr;
	torch::nn::BatchNorm bn2 = nullptr;
	torch::nn::Sequential downsample = nullptr;
	static const int64_t expansion = 1;
	int64_t stride = 1;
};
TORCH_MODULE_E(BasicBlock, NormLayer,1);
template <typename NormLayer >
struct BottleneckImpl :torch::nn::Cloneable<BottleneckImpl<NormLayer>> {
	BottleneckImpl(int64_t inplanes, int64_t planes, int64_t instride = 1, torch::nn::Sequential ds = nullptr,
		int64_t groups = 1, int64_t base_width = 64, int64_t dilation = 1) {
		int64_t width = ((int64_t)planes*(base_width / 64.0))*groups;
		conv1 = Conv1_1(inplanes, width);
		bn1 = NormLayer(width);
		conv2 = Conv3_3(width, width, instride, groups, dilation);
		bn2 = NormLayer(width);
		conv3 = Conv1_1(width, planes*expansion);
		bn3 = NormLayer(planes*expansion);
		relu = torch::nn::Functional(torch::relu);
		downsample = ds;
		stride = instride;
		register_module("conv1", conv1);
		register_module("bn1", bn1);
		register_module("conv2", conv2);
		register_module("bn2", bn2);
		register_module("conv3", conv3);
		register_module("bn3", bn3);
		register_module("relu", relu);
		if (!downsample.is_empty()) {
			register_module("downsample", downsample);
		}
	}
	torch::Tensor forward(const torch::Tensor& x) {
		auto identity = x;
		auto out = conv1(x);
		out = bn1(out);
		out = relu(out);

		out = conv2(out);
		out = bn2(out);
		out = relu(out);

		out = conv3(out);
		out = bn3(out);

		if (!downsample.is_empty()) {
			identity = downsample->forward(x);
		}
		out += identity;
		out = relu(out);
		return out;
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::Bottleneck";
	}
	torch::nn::Conv2d conv1 = nullptr;
	torch::nn::BatchNorm bn1 = nullptr;
	torch::nn::Functional relu = nullptr;
	torch::nn::Conv2d conv2 = nullptr;
	torch::nn::BatchNorm bn2 = nullptr;
	torch::nn::Conv2d conv3 = nullptr;
	torch::nn::BatchNorm bn3 = nullptr;
	torch::nn::Sequential downsample = nullptr;
	static const int64_t expansion = 4;
	int64_t stride = 1;
};
TORCH_MODULE_E(Bottleneck,NormLayer,4);
template <typename NormLayer, typename Block>
struct ResNetImpl :torch::nn::Cloneable<ResNetImpl<NormLayer,Block>> {
	ResNetImpl(int64_t* layers, int64_t num_classes = 1000, bool zero_init_residual = false,
		int64_t ingroups = 1, int64_t width_per_group = 64,
		bool* replace_stride_with_dilation = nullptr, bool only_get_layer4 = false) {
		if (replace_stride_with_dilation == nullptr) {
			bool replace_stride_with_dilations[3] = { false,false,false };
			replace_stride_with_dilation = replace_stride_with_dilations;
		}
		only_feature = only_get_layer4;
		groups = ingroups;
		base_width = width_per_group;
		conv1 = Conv2d(3, inplanes, 7, 2, 3, 1, 1, false);
		bn1 = NormLayer(inplanes);
		relu = torch::nn::Functional(torch::relu);
		maxpool = torch::nn::Functional(torch::max_pool2d, 3, 2, 1, 1, false);
		layer1 = _make_layer(64, layers[0]);
		layer2 = _make_layer(128, layers[1], 2, replace_stride_with_dilation[0]);
		layer3 = _make_layer(256, layers[2], 2, replace_stride_with_dilation[1]);
		layer4 = _make_layer(512, layers[3], 2, replace_stride_with_dilation[2]);
		if (!only_get_layer4) {
			std::vector<int64_t> a = { 1,1 };
			avgpool = torch::nn::Functional(torch::adaptive_avg_pool2d, a);
			fc = torch::nn::Linear(512 * Block::expansion, num_classes);
		}
		register_module("conv1", conv1);
		register_module("bn1", bn1);
		register_module("relu", relu);
		register_module("maxpool", maxpool);
		register_module("layer1", layer1);
		register_module("layer2", layer2);
		register_module("layer3", layer3);
		register_module("layer4", layer4);
		if (!only_get_layer4) {
			register_module("avgpool", avgpool);
			register_module("fc", fc);
		}
	}
	torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride = 1, bool dilate = false) {
		int64_t previous_dilation = dilation;
		if (dilate) {
			dilation *= stride;
			stride = 1;
		}
		torch::nn::Sequential downsample = nullptr;
		if ((stride != 1) || (inplanes != planes * Block::expansion)) {
			downsample = torch::nn::Sequential();
			downsample->push_back(Conv1_1(inplanes, planes*Block::expansion, stride));
			downsample->push_back(NormLayer(planes*Block::expansion));
		}
		ml = torch::nn::Sequential();
		ml->push_back(Block(inplanes, planes, stride, downsample,
			groups, base_width, previous_dilation));
		inplanes = planes * Block::expansion;
		for (int i = 1; i < blocks; i++) {
			ml->push_back(Block(inplanes, planes, 1, nullptr,
				groups, base_width, previous_dilation));
		}
		return ml;
	}
	torch::Tensor forward(const torch::Tensor& input) {
		auto x = conv1(input);
		x = bn1(x);
		x = relu(x);
		x = maxpool(x);
		x = layer1->forward(x);
		x = layer2->forward(x);
		x = layer3->forward(x);
		x = layer4->forward(x);
		if (!only_feature) {
			x = avgpool(x);
			x = x.view({ x.size(0),-1 });
			x = fc(x);
		}
		return x;
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::ResNet";
	}
	int64_t inplanes = 64;
	int64_t dilation = 1;
	int64_t groups = 1;
	int64_t base_width = 64;
	bool only_feature = false;
	torch::nn::Conv2d conv1 = nullptr;
	torch::nn::BatchNorm bn1 = nullptr;
	torch::nn::Functional relu = nullptr;
	torch::nn::Functional maxpool = nullptr;
	torch::nn::Sequential layer1 = nullptr;
	torch::nn::Sequential layer2 = nullptr;
	torch::nn::Sequential layer3 = nullptr;
	torch::nn::Sequential layer4 = nullptr;
	torch::nn::Functional avgpool = nullptr;
	torch::nn::Linear fc = nullptr;
	torch::nn::Sequential ml = nullptr;
};
TORCH_MODULE__(ResNet, NormLayer, Block);
template <typename NormLayer = torch::nn::BatchNorm, typename Block = Bottleneck<NormLayer>>
ResNet<NormLayer, Block> _resnetG50(bool pretrained, int64_t num_classes,
	const std::string& file_name, const std::string& name) {
	ResNet<NormLayer, Block> model = nullptr;
	if (name == "resnet101") {
		int64_t layers[4] = { 3,4,23,3 };
		model = ResNet<NormLayer, Block>(layers, num_classes);
	}
	else if (name == "resnet152") {
		int64_t layers[4] = { 3,8,36,3 };
		model = ResNet<NormLayer, Block>(layers, num_classes);
	}
	else {
		int64_t layers[4] = { 3,4,6,3 };
		model = ResNet<NormLayer, Block>(layers, num_classes);
	}
	if (pretrained) {
		printf("loading pretrained model.....");
		try {
			torch::load(model, file_name);
		}
		catch (c10::Error e) {
			printf(e.msg().c_str());
		}
	}
	return model;
}
template <typename NormLayer = torch::nn::BatchNorm, typename Block = Bottleneck<NormLayer>>
ResNet<NormLayer, Block> _resnetG50ForDV3(bool pretrained, int64_t num_classes,
	const std::string& file_name, const std::string& name) {
	ResNet<NormLayer, Block> model = nullptr;
	bool replace_stride_with_dilations[3] = { false,true,true };
	if (name == "resnet101") {
		int64_t layers[4] = { 3,4,23,3 };
		model = ResNet<NormLayer, Block>(layers, num_classes, false,
			1, 64, replace_stride_with_dilations, true);
	}
	else if (name == "resnet152") {
		int64_t layers[4] = { 3,8,36,3 };
		model = ResNet<NormLayer, Block>(layers, num_classes, false,
			1, 64, replace_stride_with_dilations, true);
	}
	else {
		int64_t layers[4] = { 3,4,6,3 };
		model = ResNet<NormLayer, Block>(layers, num_classes, false,
			1, 64, replace_stride_with_dilations, true);
	}
	if (pretrained) {
		printf("loading pretrained model.....");
		try {
			torch::load(model, file_name);
		}
		catch (c10::Error e) {
			printf(e.msg().c_str());
		}
	}
	return model;
}
template <typename NormLayer = torch::nn::BatchNorm, typename Block = BasicBlock<NormLayer>>
ResNet<NormLayer, Block> _resnetL50(bool pretrained, int64_t num_classes,
	const std::string& file_name, const std::string& name) {
	ResNet<NormLayer, Block> model = nullptr;
	if (name == "resnet34") {
		int64_t layers[4] = { 3,4,6,3 };
		model = ResNet<NormLayer, Block>(layers, num_classes);
	}
	else {
		int64_t layers[4] = { 2,2,2,2 };
		model = ResNet<NormLayer, Block>(layers, num_classes);
	}
	if (pretrained) {
		printf("loading pretrained model.....");
		try {
			torch::load(model, file_name);
		}
		catch (c10::Error e) {
			printf(e.msg().c_str());
		}
	}
	return model;
}
template<typename Backbone, typename Classifier>
struct BaseSegmentationModelImpl :torch::nn::Cloneable<BaseSegmentationModelImpl<Backbone, Classifier>> {
	BaseSegmentationModelImpl(Backbone b, Classifier c) {
		backbone = b;
		classifier = c;
		register_module("backone", b);
		register_module("classifier", c);
	}
	Backbone backbone=nullptr;
	Classifier classifier=nullptr;
	torch::Tensor forward(const torch::Tensor& input) {
		auto input_shape = input.sizes();
		input_shape = input_shape.slice(input_shape.size() - 2);
		auto features = backbone->forward(input);
		auto x = classifier->forward(features);
		x = torch::upsample_bilinear2d(x, input_shape, false);
		return x;
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::BaseSegmentationModel";
	}
};
TORCH_MODULE__(BaseSegmentationModel, Backbone, Classifier);
typedef ResNet<torch::nn::BatchNorm, Bottleneck<torch::nn::BatchNorm>> ResNetG50;
typedef ResNet<torch::nn::BatchNorm, BasicBlock<torch::nn::BatchNorm>> ResNetL50;
typedef BaseSegmentationModel<ResNetG50, DeepLabHead> SegmentationModel;

SegmentationModel DeepLabV3ResNet50(bool pretrained,int64_t num_classes,const std::string& file_name) {
	auto bt = _resnetG50ForDV3(pretrained, num_classes, file_name, "resnet50");
	auto cf = DeepLabHead(2048, num_classes);
	auto model = SegmentationModel(bt, cf);
	return model;
}
struct DeepLabV3Impl :torch::nn::Cloneable<DeepLabV3Impl> {
	DeepLabV3Impl(int64_t num_classes) {
		v3 = DeepLabV3ResNet50(false,num_classes,"");
		register_module("deeplabv3", v3);
	}
	torch::Tensor forward(const torch::Tensor& input) {
		return v3->forward(input);
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::DeepLabV3";
	}
	SegmentationModel v3 = nullptr;
};
TORCH_MODULE(DeepLabV3);
struct ResNet18Impl :torch::nn::Cloneable<ResNet18Impl> {
	ResNet18Impl(int64_t num_classes) {
		l50 = _resnetL50(false, num_classes, "", "resnet18");
		register_module("resnetl50", l50);
	}
	torch::Tensor forward(const torch::Tensor& input) {
		return l50->forward(input);
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::ResNet18";
	}
	ResNetL50 l50 = nullptr;
};
TORCH_MODULE(ResNet18);
struct ResNet34Impl :torch::nn::Cloneable<ResNet34Impl> {
	ResNet34Impl(int64_t num_classes) {
		l50 = _resnetL50(false, num_classes, "", "resnet34");
		register_module("resnetl50", l50);
	}
	torch::Tensor forward(const torch::Tensor& input) {
		return l50->forward(input);
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::ResNet34";
	}
	ResNetL50 l50 = nullptr;
};
TORCH_MODULE(ResNet34);
struct ResNet50Impl :torch::nn::Cloneable<ResNet50Impl> {
	ResNet50Impl(int64_t num_classes) {
		g50 = _resnetG50(false, num_classes, "", "resnet50");
		register_module("resnetg50", g50);
	}
	torch::Tensor forward(const torch::Tensor& input) {
		return g50->forward(input);
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::ResNet50";
	}
	ResNetG50 g50 = nullptr;
};
TORCH_MODULE(ResNet50);
struct ResNet101Impl :torch::nn::Cloneable<ResNet101Impl> {
	ResNet101Impl(int64_t num_classes) {
		g50 = _resnetG50(false, num_classes, "", "resnet101");
		register_module("resnetg50", g50);
	}
	torch::Tensor forward(const torch::Tensor& input) {
		return g50->forward(input);
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::ResNet101";
	}
	ResNetG50 g50 = nullptr;
};
TORCH_MODULE(ResNet101);
struct ResNet152Impl :torch::nn::Cloneable<ResNet152Impl> {
	ResNet152Impl(int64_t num_classes) {
		g50 = _resnetG50(false, num_classes, "", "resnet152");
		register_module("resnetg50", g50);
	}
	torch::Tensor forward(const torch::Tensor& input) {
		return g50->forward(input);
	}
	void reset()override {}
	void pretty_print(std::ostream& stream)const override {
		stream << "::ResNet152";
	}
	ResNetG50 g50 = nullptr;
};
TORCH_MODULE(ResNet152);