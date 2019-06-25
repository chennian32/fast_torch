#include <string>
#include <vector>
#include "fast_torch.h"
#include "model.h"
#include "inferencer.h"
#include "trainer.h"
#define CLASS_TYPE TrainerClass
#define SEG_TYPE TrainerSegmentation
std::shared_ptr<SEG_TYPE> _trainer = nullptr;
std::shared_ptr<CLASS_TYPE> _trainer_class = nullptr;
bool init_train(bool pretrained, bool is_class, const char* file_name,
	double momentum, double learning_rate, double weight_decay, int64_t w, int64_t h, const char* dataset_path) {
	if (is_class) {
		_trainer = nullptr;
		_trainer_class = std::make_shared<CLASS_TYPE>(pretrained, file_name, momentum, learning_rate,
			weight_decay, w, h, dataset_path);
		return _trainer_class->is_valid();
	}
	else {
		_trainer_class = nullptr;
		_trainer = std::make_shared<SEG_TYPE>(pretrained, file_name, momentum, learning_rate,
			weight_decay, w, h, dataset_path);
		return _trainer->is_valid();
	}
	return false;
}
void exit_train(){
	if (_trainer != nullptr) {
		_trainer = nullptr;
	}
	if (_trainer_class != nullptr) {
		_trainer_class = nullptr;
	}
}
bool train(int64_t epoch_count, const char* model_save_path,
	const char* device_type, int64_t eval_count, int64_t lr_decay_count, double lr_decay_scale,
	callback callback, int64_t batch_size, int64_t workers){
	bool ret = false;
	if (_trainer != nullptr) {
		ret=_trainer->train(epoch_count, model_save_path, device_type, eval_count, lr_decay_count, lr_decay_scale,
			callback, batch_size, workers);
	}
	if (_trainer_class != nullptr) {
		ret=_trainer_class->train(epoch_count, model_save_path, device_type, eval_count, lr_decay_count, lr_decay_scale,
			callback, batch_size, workers);
	}
	return ret;
}
#define INF_TYPE Inferencer
#define INF_CLASS_TYPE InferencerClass
std::map < std::string, std::shared_ptr<INF_TYPE>> _inferencers;
std::map < std::string, std::shared_ptr<INF_CLASS_TYPE>> _inferencer_classes;
bool init_inference(const char* file_name,int64_t num_classes,const char* name, const char* device_string,bool is_class){
	bool ret = false;
	if (is_class) {
		auto f = _inferencer_classes.find(name);
		if (f == _inferencer_classes.end()) {
			auto i = std::make_shared<INF_CLASS_TYPE>(num_classes, file_name, device_string);
			_inferencer_classes.insert(std::pair<std::string, std::shared_ptr<INF_CLASS_TYPE>>(name, i));
			ret = i->is_valid();
		}
		else {
			if (f->second != nullptr)
				ret = f->second->is_valid();
		}
	}
	else {
		auto f = _inferencers.find(name);
		if (f == _inferencers.end()) {
			auto i = std::make_shared<INF_TYPE>(num_classes, file_name, device_string);
			_inferencers.insert(std::pair<std::string, std::shared_ptr<INF_TYPE>>(name, i));
			ret = i->is_valid();
		}
		else {
			if(f->second!=nullptr)
				ret = f->second->is_valid();
		}
	}
	return ret;
}
void exit_inference(const char* name,bool is_class){
	if (is_class) {
		auto f = _inferencer_classes.find(name);
		if (f == _inferencer_classes.end())
			return;
		f->second.reset();
		_inferencer_classes.erase(f);
	}
	else {
		auto f = _inferencers.find(name);
		if (f == _inferencers.end())
			return;
		f->second.reset();
		_inferencers.erase(f);
	}
}
bool inference_class_file(const char* name, int64_t w, int64_t h, const char*  file_name,
	int64_t topk,float* score,int64_t * out){
	auto f = _inferencer_classes.find(name);
	if (f == _inferencer_classes.end())
		return false;
	if (f->second == nullptr)
		return false;
	if (!f->second->inference_file(file_name, w, h))
		return false;
	if (!f->second->get_result(topk, out, score))
		return false;
	return true;
}
bool inference_class(const char* name,int64_t w, int64_t h,void* ptr, bool is_rgba,
	int64_t topk, float* score, int64_t * out){
	auto f = _inferencer_classes.find(name);
	if (f == _inferencer_classes.end())
		return false;
	if (f->second == nullptr)
		return false;
	if (!f->second->inference(w,h,ptr,is_rgba))
		return false;
	if (!f->second->get_result(topk, out, score))
		return false;
	return true;
}
bool inference_file(const char* name, int64_t w, int64_t h, const char*  file_name,
	float* score, uint8_t * out){
	auto f = _inferencers.find(name);
	if (f == _inferencers.end())
		return false;
	if (f->second == nullptr)
		return false;
	if (!f->second->inference_file(file_name,w, h))
		return false;
	if (!f->second->get_result(out, score))
		return false;
	return true;
}
bool inference(const char* name, int64_t w, int64_t h, void* ptr, bool is_rgba,
	 float* score, uint8_t * out){
	auto f = _inferencers.find(name);
	if (f == _inferencers.end())
		return false;
	if (f->second == nullptr)
		return false;
	if (!f->second->inference(w, h, ptr, is_rgba))
		return false;
	if (!f->second->get_result(out, score))
		return false;
	return true;
}
bool get_inference_lass_error(const char* name, char* error_msg, int64_t& size)
{
	auto f = _inferencers.find(name);
	if (f == _inferencers.end())
		return false;
	auto s = f->second->last_error();
	size = s.size();
	if (error_msg == nullptr)
		return true;
	memcpy(error_msg, s.c_str(), size);
	return true;
}
bool get_inference_class_lass_error(const char* name, char* error_msg, int64_t& size)
{
	auto f = _inferencer_classes.find(name);
	if (f == _inferencer_classes.end())
		return false;
	auto s=f->second->last_error();
	size = s.size();
	if (error_msg == nullptr)
		return true;
	memcpy(error_msg, s.c_str(), size);
	return true;
}
bool get_train_lass_error(char* error_msg, int64_t& size)
{
	if (_trainer == nullptr)
		return false;
	auto s = _trainer->last_error();
	size = s.size();
	if (error_msg == nullptr)
		return true;
	memcpy(error_msg, s.c_str(), size);
	return true;
}
bool get_train_class_lass_error(char* error_msg, int64_t& size)
{
	if (_trainer_class == nullptr)
		return false;
	auto s = _trainer_class->last_error();
	size = s.size();
	if (error_msg == nullptr)
		return true;
	memcpy(error_msg, s.c_str(), size);
	return true;
}
void test_train(){
	std::string datasetPath = "D:\\dataset\\ADEChallengeData2017";
	//std::string datasetPath = "D:\\case\\testclass\\";
	std::string modelPath = "D:\\m3.pt";
	if (!init_train(true,false, modelPath.c_str(), 0.9, 0.001, 0.00005, 512, 512, datasetPath.c_str()))
		return;
	train(100, modelPath.c_str(), "cuda:0", 500, 10, 0.1, nullptr,6, 0);
	exit_train();
}
void test_class_train()
{
	std::string datasetPath = "D:\\case\\pill\\";
	//std::string datasetPath = "D:\\dataset\\imgnetval\\";
	std::string modelPath = "D:\\case\\xizhu\\__zb__\\trained_class_.pt";
	if (!init_train(true, true, modelPath.c_str(), 0.9, 0.001, 0.00001,224,224, datasetPath.c_str()))
		return;
	train(100, modelPath.c_str(), "cuda:0", 100, 30, 0.1, nullptr, 64, 0);
	exit_train();
}
void test_inference(){
	if (!init_inference("D:\\m1.pt",5, "test1", "cuda:0",false))
		return;
	float* s = new float[256 * 256];
	uint8_t* o = new uint8_t[256 * 256];
	inference_file("test1", 256, 256, "D:\\case\\testclass\\images\\pill_ginseng_contamination_003.png",s,o);
	cv::Mat mat(256, 256, CV_8UC1, o);
	cv::imwrite("D:\\0.bmp", mat);
	exit_inference("test1",false);
}
void test_inference_class() {
	if (!init_inference("D:\\case\\xizhu\\__zb__\\trained_class.pt", 3, "test1", "cuda:0", true))
		return;
	float* s = new float[3 * 3];
	int64_t* o = new int64_t[3 * 3];
	inference_class_file("test1", 224, 224, "D:\\case\\testclass\\contamination\\pill_ginseng_contamination_001.png",3, s, o);
	printf("%f", s[0]);
	printf("%f", s[1]);
	printf("%f", s[2]);
	printf(",,%d", o[0]);
	printf(",,%d", o[1]);
	printf(",,%d", o[2]);
	exit_inference("test1", true);
	delete[] s;
	delete[] o;
}
int main(){
	//test_class_train_pre();
	//test_inference_class();
	//eval_inference_dataset_class();
	test_train();
	//test_class_train();
	return -1;
}