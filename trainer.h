#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "tensor.h"
#include "model.h"
#include "file_util.h"
#include "optimizer.h"
template <typename Module, typename Dataset>
class TrainerBase{
private:
	Module model_=nullptr;
	std::shared_ptr<SGD> optimizer_;
	bool is_valid_;
	std::string last_error_;
	bool _is_exit_;
	std::shared_ptr<Dataset> dataset_;
	int64_t width_;
	std::shared_ptr<Dataset> dataset_eval_;
	int64_t height_;
	bool is_class_;
protected:
	bool compute_iou(torch::Tensor& pred, torch::Tensor& label,float& iou) {
		pred = torch::argmax(pred, 0);
		auto classes = std::get<0>(torch::_unique2(label));
		float ious = 0;
		int64_t size = classes.numel();
		int64_t idx = 0;
		for (int64_t i = 0; i < size; i++) {
			auto c = classes[i].item().toLong();
			if (c == 0) 			
				continue;
			idx++;
			auto pred_c = pred == c;
			auto label_c = label == c;
			auto intersectionR = torch::__and__(pred_c, label_c).sum().item().toLong();
			auto unionR = torch::__or__(pred_c, label_c).sum().item().toLong();
			auto isEq = unionR == 0;
			auto d = isEq ? unionR : intersectionR*1.0f / unionR;
			ious += d;
		}
		if (idx == 0) {
			auto classes_pred=std::get<0>(torch::_unique2(pred));
			if (classes_pred.numel() == 1) {
				if (classes[0].item().toLong() == 0) {
					return false;
				}
			}
			else
				idx = 1;
		}
		iou = ious / idx;
		return true;
	}
	float compute_iou_batch(torch::Tensor& preds, torch::Tensor&labels) {
		int64_t batch = preds.size(0);
		float mean = 0;
		int64_t index = 0;
		for (int64_t b = 0; b < batch; b++) {
			float iou = 0;
			if (compute_iou(preds[b], labels[b], iou)) {
				mean += iou;
				index += 1;
			}
		}
		auto iou_out = index == 0 ? 0.0f: (mean / index);
		return iou_out;
	}
	float compute_class_pred_batch(torch::Tensor& pred, torch::Tensor& target) {
		auto max = pred.max(1);
		auto pred_max = std::get<1>(max);
		auto size=pred_max.numel();
		auto tsq = target.squeeze();
		float target_sum = pred_max.eq(tsq).sum().item<int64_t>()*1.0;
		return target_sum / size;
	}
	virtual float get_prediction(torch::Tensor& pred, torch::Tensor& target) {
		if (is_class_)
			return compute_class_pred_batch(pred, target);
		return	compute_iou_batch(pred, target);
	}
	virtual torch::Tensor get_loss(torch::Tensor& pred, torch::Tensor& target)
	{
		auto pred_log = torch::log_softmax(pred, 1);
		if (is_class_)
			return torch::nll_loss(pred_log, target.squeeze());
		return torch::nll_loss2d(pred_log, target);
	}
public:
	explicit TrainerBase(bool pretrained, const std::string& model_file_name,
		double momentum, double learning_rate, double weight_decay, int64_t w,int64_t h,
		const char* dataset_path,bool is_class):width_(w),height_(h){
		is_valid_ = false;
		_is_exit_ = false;
		dataset_ = std::make_shared<Dataset>(true, w, h);
		if (!dataset_->init_dataset(dataset_path))
			return;		
		dataset_eval_=dataset_->split();
		is_class_ = is_class;
		try {
			model_ = UsedNet<Module>(pretrained, dataset_->get_class_count(), model_file_name);
			optimizer_ = std::make_shared<SGD>(model_->parameters(),
				SGDOptions(learning_rate).momentum(momentum).weight_decay(weight_decay));
			is_valid_ = true;
		}
		catch (c10::Error e) {
			last_error_ = e.msg();
		}
	}
	virtual ~TrainerBase(){	 
		model_ = nullptr;
		dataset_ = nullptr;
		dataset_eval_ = nullptr;
		is_valid_ = false;
		_is_exit_ = true;
	}
	bool train(int64_t epoch_count, const std::string& model_save_path, const std::string& device_type, int64_t eval_count,
		int64_t lr_decay_count, double lr_decay_scale, callback callback,int64_t batch_size, int64_t workers);
	bool eval(const std::string& device_type, const std::string& model_save_path, int64_t step, int64_t all_step,
		callback callback, int64_t batch_size,int64_t workers,float max_pred);
	bool is_valid()const { return is_valid_; }
	std::string last_error()const { return last_error_; }
	void exit_train() { _is_exit_ = true; }
};
template <typename Module, typename Dataset>
bool TrainerBase<Module, Dataset>::eval(const std::string& device_type, const std::string& model_save_path, 
	int64_t step,int64_t all_step,callback callback, int64_t batch_size, int64_t workers, float max_pred) {
	if (!is_valid_)
		return false;
	torch::manual_seed(1);
	auto map_dataset = dataset_eval_->map(torch::data::transforms::Stack<>());
	try {
		auto dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(map_dataset),
			torch::data::DataLoaderOptions(batch_size).workers(workers).drop_last(true));
		torch::Device d(device_type);
		model_->to(d);
		int64_t idx = 0;
		float preds = 0;
		for (auto& batch : *dataloader) {
			if (_is_exit_)
				return false;
			auto data = batch.data.to(d), targets = batch.target.to(d);
			torch::NoGradGuard no_grad;
			model_->eval();
			auto prediction = model_->forward(data);			
			float pred = get_prediction(prediction, targets);
			idx++;
			preds += pred;
		}
		auto mean = idx == 0 ? 0.0f : preds / idx;
		if (callback == nullptr)
			std::cout << "Epoch: " << 0 << " | Batch: " << 0
			<< " | Loss: " << 0 << " | Prediction: " << mean << std::endl;
		else
			callback(0, mean, 0, step, all_step, 0);
		if (mean > max_pred) {
			torch::save(model_, model_save_path);
			max_pred = mean;
		}
		dataloader.release();
	}
	catch (torch::Error e) {
		last_error_ = e.msg();
		printf(last_error_.c_str());
		return false;
	}
	catch (std::runtime_error& e) {
		last_error_ = e.what();
		printf(last_error_.c_str());
		return false;
	}
	return true;
}
#define TRY_CATCH
template <typename Module, typename Dataset>
bool TrainerBase<Module, Dataset>::train(int64_t epoch_count, const std::string& model_save_path,
	const std::string& device_type, int64_t eval_count, int64_t lr_decay_count, double lr_decay_scale,
	callback callback, int64_t batch_size, int64_t workers){
	if (!is_valid_)
		return false;
	torch::manual_seed(1);
	auto map_dataset = dataset_->map(torch::data::transforms::Stack<>());
	auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(map_dataset),
		torch::data::DataLoaderOptions(batch_size).workers(workers).drop_last(true));
#ifdef TRY_CATCH
	try {
#endif
		auto length = dataset_->size().value();
		auto batch_count = length / batch_size;
		auto all_count = batch_count * epoch_count;
		auto all_step = all_count / eval_count;
		torch::Device d(device_type);
		model_->to(d);
		float max = FLT_MIN;
		int64_t batch_index = 0;
		double lr = optimizer_->options.learning_rate();
		int64_t step=0;
		for (size_t epoch = 0; epoch < epoch_count; ++epoch) {
			for (auto& batch : *dataloader) {
				if (_is_exit_)
					return false;
				model_->train();				
				auto data = batch.data.to(d), targets = batch.target.to(d);		
				optimizer_->zero_grad();
				auto prediction = model_->forward(data);
				auto loss = get_loss(prediction, targets);	
				loss.backward();		
				optimizer_->step();
				float lossIn = loss.item<float>();
				lr = optimizer_->options.learning_rate();
				if (callback == nullptr)
					std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
					<< " | Loss: " << lossIn << " | LearningRate: " << lr << std::endl;
				else
					callback(lossIn, FLT_MAX, epoch, batch_index,all_count, lr);
				if (++batch_index % eval_count == 0) {
					eval(device_type, model_save_path, step, all_step, callback, batch_size, workers, max);
					step++;
				}
			}
			if ((epoch%lr_decay_count == 0) && epoch != 0) {
				lr = lr * lr_decay_scale;
				optimizer_->options.learning_rate(lr);
			}
		}
		auto dir = file_util::get_dir_name(model_save_path);
		torch::save(model_, dir + "\\last.pt");
#ifdef TRY_CATCH
	}
	catch (torch::Error e) {
		last_error_ = e.msg();
		printf(last_error_.c_str());
		return false;
	}
	catch (std::runtime_error& e) {
		last_error_ = e.what();
		printf(last_error_.c_str());
		return false;
	}
	dataloader.release();
#endif
	return true;
}
template <typename Module, typename Dataset>
class TrainerSegmentationBase :public TrainerBase<Module,Dataset> {
public:
	explicit TrainerSegmentationBase(bool pretrained, const std::string& model_file_name,
		double momentum, double learning_rate, double weight_decay, int64_t w, int64_t h,
		const char* dataset_path) :
		TrainerBase<Module, Dataset>(pretrained, model_file_name, momentum,
			learning_rate, weight_decay, w, h, dataset_path, false) {}
};
template <typename Module, typename Dataset>
class TrainerClassBase :public TrainerBase<Module, Dataset> {
public:
	explicit TrainerClassBase(bool pretrained, const std::string& model_file_name,
		double momentum, double learning_rate, double weight_decay, int64_t w, int64_t h,
		const char* dataset_path) :
		TrainerBase<Module, Dataset>(pretrained, model_file_name, momentum,
			learning_rate, weight_decay, w, h, dataset_path, true) {}
};
class TrainerSegmentation :public TrainerSegmentationBase<XceptionAx3, CustomDataset>{
public:
	explicit TrainerSegmentation(bool pretrained, const std::string& model_file_name,
		double momentum, double learning_rate, double weight_decay,int64_t w, int64_t h,
		const char* dataset_path) :
		TrainerSegmentationBase<XceptionAx3, CustomDataset>(pretrained, model_file_name, momentum,
			learning_rate, weight_decay,w,h,dataset_path){}
};
class TrainerClass :public TrainerClassBase<XceptionA, CustomClassDataset>{
public:
	explicit TrainerClass(bool pretrained,const std::string& model_file_name,
		double momentum, double learning_rate, double weight_decay, int64_t w, int64_t h,const char* dataset_path) :
		TrainerClassBase<XceptionA, CustomClassDataset>(pretrained, model_file_name, momentum,
			learning_rate, weight_decay,w,h,dataset_path){}
};
class TrainerDeeplabV3 :public TrainerSegmentationBase <DeepLabV3 , CustomDataset > {
public:
	explicit TrainerDeeplabV3(bool pretrained, const std::string& model_file_name,
		double momentum, double learning_rate, double weight_decay, int64_t w, int64_t h,
		const char* dataset_path) :
		TrainerSegmentationBase<DeepLabV3, CustomDataset>(pretrained, model_file_name, momentum,
			learning_rate, weight_decay, w, h, dataset_path) {}
};
class TrainerResNet :public TrainerClassBase<ResNet50, ImageNetValDataset> {
public:
	explicit TrainerResNet(bool pretrained, const std::string& model_file_name,
		double momentum, double learning_rate, double weight_decay, int64_t w, int64_t h, const char* dataset_path) :
		TrainerClassBase<ResNet50, ImageNetValDataset>(pretrained, model_file_name, momentum,
			learning_rate, weight_decay, w, h, dataset_path) {}
};
class TrainerXception :public TrainerClassBase<XceptionA, ImageNetValDataset> {
public:
	explicit TrainerXception(bool pretrained, const std::string& model_file_name,
		double momentum, double learning_rate, double weight_decay, int64_t w, int64_t h, const char* dataset_path) :
		TrainerClassBase<XceptionA, ImageNetValDataset>(pretrained, model_file_name, momentum,
			learning_rate, weight_decay, w, h, dataset_path) {}
};
class TrainerDFANet :public TrainerSegmentationBase <XceptionAx3, ADKDataset >{
public:
	explicit TrainerDFANet(bool pretrained, const std::string& model_file_name,
		double momentum, double learning_rate, double weight_decay, int64_t w, int64_t h,
		const char* dataset_path) :
		TrainerSegmentationBase<XceptionAx3, ADKDataset>(pretrained, model_file_name, momentum,
			learning_rate, weight_decay, w, h, dataset_path) {}
};