#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "tensor.h"
#include "model.h"
#include "file_util.h"
template <typename Module>
class InferencerBase
{
private:
	Module model_=nullptr;
	std::string device_;
	bool is_valid_;
protected:
	torch::Tensor output_;
	int64_t width_last_;
	int64_t height_last_;
	bool last_result_;
	std::string last_error_;
public:
	explicit InferencerBase(int64_t num_classes, const std::string& model_file_name, const std::string& device) {
		is_valid_ = false;
		try {
			model_ = UsedNet<Module>(true, num_classes, model_file_name);
			device_ = device;
			model_->to(device_);
			is_valid_ = true;
		}
		catch (c10::Error e) {
			last_error_ = e.msg();
		}
	}
	virtual ~InferencerBase() {
		model_ = nullptr;
		is_valid_ = false;
		last_result_ = false;
	}
	bool inference_file(const std::string& image_file, int64_t w, int64_t h) {
		try {
			width_last_ = w;
			height_last_ = h;
			last_result_ = false;
			if (!is_valid_)
				return false;
			torch::Tensor t;
			if (!tensor_load::load(image_file, cv::Size(w, h), t, false,0))
				return false;
			t = t.unsqueeze(0);
			t = t.to(torch::Device(device_));
			{
				torch::NoGradGuard ng;
				model_->eval();
				output_ = model_->forward(t);
			}
			last_result_ = true;
			return true;
		}
		catch (c10::Error e) {
			last_error_ = e.msg();;
		}
		catch (...) {
		}
		return false;
	}
	bool inference(int64_t w, int64_t h, void* ptr, bool is_rgba) {
		try {
			width_last_ = w;
			height_last_ = h;
			last_result_ = false;
			if (!is_valid_)
				return false;
			torch::Tensor t;
			if (is_rgba) {
				if (!tensor_load::load_rgba(ptr, w, h, t))
					return false;
			}
			else {
				if (!tensor_load::load_rgb(ptr, w, h, t))
					return false;
			}
			t = t.unsqueeze(0);
			t = t.to(torch::Device(device_));
			{
				torch::NoGradGuard ng;
				model_->eval();
				output_ = model_->forward(t);
			}
			last_result_ = true;
			return true;
		}
		catch (torch::Error e) {
			last_error_ = e.msg();
		}
		catch (...) {
		}
		return false;
	}
	bool is_valid()const { return is_valid_; }
	std::string last_error()const { return last_error_; }
	std::string device()const { return device_; }
};
class Inferencer :public InferencerBase<XceptionAx3> {
public:
	explicit Inferencer(int64_t num_classes, const std::string& model_file_name, const std::string& device) :
		InferencerBase<XceptionAx3>(num_classes, model_file_name, device) {}
	virtual ~Inferencer() {}
	//out size={mWidthLast,mHeighLast,long)
	//out_score={mWidthLast,mHeighLast,float}
	bool get_result(void* out, void* out_score) {
		if (!last_result_)
			return false;
		try {
			auto all = torch::max(output_, 1);
			auto out_ = std::get<1>(all).squeeze();
			auto out_score_ = std::get<0>(all).squeeze();
			//out_ = out_.permute({ 1,0 });
			out_ = out_.to(torch::kUInt8).cpu();
			//out_score_ = out_score_.permute({ 1,0 });
			out_score_ = out_score_.cpu();
			memcpy(out, out_.data_ptr(), out_.nbytes());
			memcpy(out_score, out_score_.data_ptr(), out_score_.nbytes());
		}
		catch (c10::Error e) {
			last_error_ = e.msg();
			return false;
		}
		return true;
	}
};
class InferencerClass :public InferencerBase<XceptionA> {
public:
	explicit InferencerClass(int64_t num_classes, const std::string& model_file_name, const std::string& device) :
		InferencerBase<XceptionA>(num_classes, model_file_name, device) {}
	virtual ~InferencerClass() {}
	//out size={mWidthLast,mHeighLast,long)
	//out_score={mWidthLast,mHeighLast,float}
	bool get_result(int64_t topk, int64_t * out, float* out_score) {
		if (!last_result_)
			return false;
		try {		
			output_ = output_.squeeze().cpu();
			auto tuple = torch::topk(output_, topk);
			auto value = std::get<0>(tuple);
			value = value.softmax(0);
			auto index = std::get<1>(tuple);
			memcpy(out_score, value.data_ptr(), value.nbytes());
			memcpy(out, index.data_ptr(), index.nbytes());
		}
		catch (c10::Error e) {
			last_error_ = e.msg();
			return false;
		}
		return true;
	}
};
class InferencerJitBase
{
private:
	std::shared_ptr<torch::jit::script::Module> model_;
	std::string device_;
	bool is_valid_;
protected:
	torch::Tensor output_;
	int64_t width_last_;
	int64_t height_last_;
	bool last_result_;
	std::string last_error_;
public:
	explicit InferencerJitBase(const std::string& model_file_name, const std::string& device) {
		is_valid_ = false;
		try {
			model_ = torch::jit::load(model_file_name,torch::Device(device));
			device_ = device;
			is_valid_ = true;
		}
		catch (c10::Error e) {
			last_error_ = e.msg();
		}
	}
	virtual ~InferencerJitBase() {
		model_ = nullptr;
		is_valid_ = false;
		last_result_ = false;
	}
	bool inference_file(const std::string& image_file, int64_t w, int64_t h) {
		try {
			width_last_ = w;
			height_last_ = h;
			last_result_ = false;
			if (!is_valid_)
				return false;
			torch::Tensor t;
			if (!tensor_load::load(image_file, cv::Size(w, h), t, false, 0))
				return false;
			t = t.unsqueeze(0);
			t = t.to(torch::Device(device_));
			{
				torch::NoGradGuard ng;
				model_->eval();
				output_ = model_->forward({ t }).toTensor();
			}
			last_result_ = true;
			return true;
		}
		catch (c10::Error e) {
			last_error_ = e.msg();;
		}
		catch (...) {
		}
		return false;
	}
	bool inference(int64_t w, int64_t h, void* ptr, bool is_rgba) {
		try {
			width_last_ = w;
			height_last_ = h;
			last_result_ = false;
			if (!is_valid_)
				return false;
			torch::Tensor t;
			if (is_rgba) {
				if (!tensor_load::load_rgba(ptr, w, h, t))
					return false;
			}
			else {
				if (!tensor_load::load_rgb(ptr, w, h, t))
					return false;
			}
			t = t.unsqueeze(0);
			t = t.to(torch::Device(device_));
			{
				torch::NoGradGuard ng;
				model_->eval();
				output_ = model_->forward({ t }).toTensor();
			}
			last_result_ = true;
			return true;
		}
		catch (torch::Error e) {
			last_error_ = e.msg();
		}
		catch (...) {
		}
		return false;
	}
	bool is_valid()const { return is_valid_; }
	std::string last_error()const { return last_error_; }
	std::string device()const { return device_; }
};
class InferencerJit :public InferencerJitBase
{
public:
	explicit InferencerJit(int64_t num_classes, const std::string& model_file_name, const std::string& device) :
		InferencerJitBase(model_file_name, device) {}
	virtual ~InferencerJit() {}
	bool get_result(void* out, void* out_score) {
		if (!last_result_)
			return false;
		try {
			auto all = torch::max(output_, 1);
			auto out_ = std::get<1>(all).squeeze();
			auto out_score_ = std::get<0>(all).squeeze();
			//out_ = out_.permute({ 1,0 });
			out_ = out_.to(torch::kUInt8).cpu();
			//out_score_ = out_score_.permute({ 1,0 });
			out_score_ = out_score_.cpu();
			memcpy(out, out_.data_ptr(), out_.nbytes());
			memcpy(out_score, out_score_.data_ptr(), out_score_.nbytes());
		}
		catch (c10::Error e) {
			last_error_ = e.msg();
			return false;
		}
		return true;
	}
};
class InferencerJitClass :public InferencerJitBase
{
public:
	explicit InferencerJitClass(int64_t num_classes,const std::string& model_file_name, const std::string& device) :
		InferencerJitBase(model_file_name, device) {}
	virtual ~InferencerJitClass() {}
	//out size={mWidthLast,mHeighLast,long)
	//out_score={mWidthLast,mHeighLast,float}
	bool get_result(int64_t topk, int64_t * out, float* out_score) {
		if (!last_result_)
			return false;
		try {
			output_ = output_.squeeze().cpu();
			auto tuple = torch::topk(output_, topk);
			auto value = std::get<0>(tuple);
			value = value.softmax(0);
			auto index = std::get<1>(tuple);
			memcpy(out_score, value.data_ptr(), value.nbytes());
			memcpy(out, index.data_ptr(), index.nbytes());
		}
		catch (c10::Error e) {
			last_error_ = e.msg();
			return false;
		}
		return true;
	}
};