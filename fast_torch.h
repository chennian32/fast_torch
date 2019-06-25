#pragma once
extern "C" {
	typedef void(*callback)(float loss, float pred, int64_t epoch_index, int64_t batch_index,int64_t batch_count,float lr);
	__declspec(dllexport) bool init_train(bool pretrained, bool is_class, const char* file_name,
		double momentum, double learning_rate, double weight_decay, int64_t w, int64_t h, const char* dataset_path);
	__declspec(dllexport) bool train(int64_t epoch_count, const char* model_save_path,
		const char* device_type,int64_t eval_count, int64_t lr_decay_count, double lr_decay_scale,
		callback callback,int64_t batch_size, int64_t workers);
	__declspec(dllexport) void exit_train();
	__declspec(dllexport) bool init_inference(const char* file_name, int64_t num_classes, const char* name,
		const char* device_string, bool is_class);
	__declspec(dllexport)  void exit_inference(const char* name,bool is_class);
	__declspec(dllexport) bool inference_class_file(const char* name, int64_t w, int64_t h, const char*  file_name,
		int64_t topk, float* score, int64_t * out);
	__declspec(dllexport) bool inference_class(const char* name, int64_t w, int64_t h, void* ptr, bool is_rgba,
		int64_t topk, float* score, int64_t * out);
	__declspec(dllexport) bool inference_file(const char* name, int64_t w, int64_t h, const char*  file_name,
		float* score, uint8_t * out);
	__declspec(dllexport) bool inference(const char* name, int64_t w, int64_t h, void* ptr, bool is_rgba,
		float* score, uint8_t * out);
	__declspec(dllexport) bool get_inference_lass_error(const char* name,char* error_msg, int64_t& size);
	__declspec(dllexport) bool get_inference_class_lass_error(const char* name, char* error_msg, int64_t& size);
	__declspec(dllexport) bool get_train_lass_error(char* error_msg, int64_t& size);
	__declspec(dllexport) bool get_train_class_lass_error(char* error_msg, int64_t& size);
}
