#include "tensor.h"
#include "file_util.h"
bool tensor_load::load(std::string file_name, cv::Size size, torch::Tensor &input_tensor, bool flip, int ro) {
	cv::Mat  image = cv::imread(file_name);
	if (image.empty() || !image.data)
		return false;
	int64_t c = image.channels();
	if (c != 3)
		return false;
	cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2RGB);
	if (flip)
		cv::flip(image, image, 1);
	if (ro == 180)
		cv::rotate(image, image, cv::RotateFlags::ROTATE_180);
	else if (ro == 90)
		cv::rotate(image, image, cv::RotateFlags::ROTATE_90_CLOCKWISE);
	else if (ro == 270)
		cv::rotate(image, image, cv::RotateFlags::ROTATE_90_COUNTERCLOCKWISE);
	cv::resize(image, image, size);
	image.convertTo(image, CV_32FC3, 1.0 / 255.0);
	torch::TensorOptions option(torch::kFloat32);
	auto img_tensor = torch::from_blob(
		image.data, { image.rows, image.cols, image.channels() }, option);
	img_tensor = img_tensor.permute({ 2,0,1 }).contiguous();
	img_tensor[0] = img_tensor[0].sub_(0.485).div_(0.229);
	img_tensor[1] = img_tensor[1].sub_(0.456).div_(0.224);
	img_tensor[2] = img_tensor[2].sub_(0.406).div_(0.225);
	input_tensor = img_tensor.clone();
	return true;
}
bool tensor_load::load_seg(std::string file_name,cv::Size size, torch::Tensor &input_tensor,bool flip,int ro){
	cv::Mat  image = cv::imread(file_name, cv::ImreadModes::IMREAD_GRAYSCALE);
	if (image.empty() || !image.data) 
		return false;
	if (image.channels() >1)
		return false;
	if (flip)
		cv::flip(image, image, 1);
	if (ro == 180)
		cv::rotate(image, image, cv::RotateFlags::ROTATE_180);
	else if (ro == 90) 
		cv::rotate(image, image, cv::RotateFlags::ROTATE_90_CLOCKWISE);
	else if (ro == 270)
		cv::rotate(image, image, cv::RotateFlags::ROTATE_90_COUNTERCLOCKWISE);
	cv::resize(image, image, size, 0, 0, cv::InterpolationFlags::INTER_NEAREST);
	torch::TensorOptions option(torch::kByte);
	auto img_tensor = torch::from_blob(
		image.data, { image.rows, image.cols }, option);
	input_tensor = img_tensor.toType(torch::kLong);
	return true;
}
bool  tensor_load::load_rgb(void* ptr, int64_t w, int64_t h,  torch::Tensor &input_tensor){
	cv::Mat image(cv::Size(w, h), CV_8UC3, ptr);
	if (image.empty() || !image.data)
		return false;
	image.convertTo(image, CV_32FC3, 1.0 / 255.0);
	torch::TensorOptions option(torch::kFloat32);
	auto img_tensor = torch::from_blob(
		image.data, { image.rows, image.cols, image.channels() }, option);
	img_tensor = img_tensor.permute({ 2,0,1 });
	img_tensor[0] = img_tensor[0].sub_(0.485).div_(0.229);
	img_tensor[1] = img_tensor[1].sub_(0.456).div_(0.224);
	img_tensor[2] = img_tensor[2].sub_(0.406).div_(0.225);
	input_tensor = img_tensor.clone();
	return true;
}
bool  tensor_load::load_rgba(void* ptr, int64_t w, int64_t h, torch::Tensor &input_tensor){
	std::vector<cv::Mat> mvs;
	cv::Mat image(cv::Size(w, h), CV_8UC4, ptr);
	if (image.empty() || !image.data)
		return false;
	cv::split(image, mvs);
	mvs.pop_back();
	cv::merge(mvs, image);
	image.convertTo(image, CV_32FC3, 1.0 / 255.0);
	torch::TensorOptions option(torch::kFloat32);
	auto img_tensor = torch::from_blob(
		image.data, { image.rows, image.cols, image.channels() }, option);
	img_tensor = img_tensor.permute({ 2,0,1 });
	img_tensor[0] = img_tensor[0].sub_(0.485).div_(0.229);
	img_tensor[1] = img_tensor[1].sub_(0.456).div_(0.224);
	img_tensor[2] = img_tensor[2].sub_(0.406).div_(0.225);
	input_tensor = img_tensor.clone();
	return true;
}
bool CustomClassDataset::init_dataset(const std::string& root){
	auto dirs = file_util::list_dirs(root);
	int64_t idx = -1;
	for (auto dir : dirs){	
		if (file_util::is_dir(dir)) {
			auto file_name = file_util::get_file_name_without_ext(dir);
			auto files = file_util::list_files(dir);
			for (auto file : files) {
				if (_zables.find(file_name) == _zables.end()) {
					_zables.insert(file_name);
					idx++;
				}
				_image_class_files.push_back(std::pair<std::string, int64_t>(file, idx));
			}
		}
	}
	_class_count = _zables.size();
	if ((_image_class_files.size() == 0)||_class_count==0)
		return false;
	return true;
}
bool CustomDataset::init_dataset(const std::string& root) {
	auto dirs_and_files = file_util::list_files_and_dirs(root);
	std::vector<std::string> image_files;
	std::vector<std::string> seg_files;
	for (auto dir_or_file : dirs_and_files){
		if (file_util::is_dir(dir_or_file)) {
			auto file_name = file_util::get_file_name_without_ext(dir_or_file);
			bool is_image = file_name == "images";
			bool is_seg = file_name == "segs";
			auto files = file_util::list_files(dir_or_file);
			for (auto file : files) {
				if (is_image)
					image_files.push_back(file);
				else if (is_seg)
					seg_files.push_back(file);
			}
		}
		else if (file_util::get_file_name(dir_or_file) == "zable.txt") {
			std::string read = file_util::read_file_text(dir_or_file.c_str());
			read=file_util::to_local(read);
			_zables = file_util::split(read, "\n");
		}
	}
	_class_count = _zables.size();
	if (seg_files.size() != image_files.size())
		return false;
	if ((image_files.size() == 0) || _class_count == 0)
		return false;
	for (int64_t i = 0; i < image_files.size(); i++) {
		_image_seg_files.push_back(std::pair<std::string, std::string>(image_files[i], seg_files[i]));
	}
	return true;
}
bool ImageNetValDataset::init_dataset(const std::string& root) {
	auto dirs_and_files = file_util::list_files_and_dirs(root);
	std::vector<std::string> image_files;
	std::vector<int64_t> image_classes;
	for (auto dir_or_file : dirs_and_files) {
		if (file_util::is_dir(dir_or_file)) {
			auto file_name = file_util::get_file_name_without_ext(dir_or_file);
			bool is_image = file_name == "ILSVRC2012_img_val";
			auto files = file_util::list_files(dir_or_file);
			for (auto file : files) {
				if (is_image)
					image_files.push_back(file);
			}
		}
		else if (file_util::get_file_name(dir_or_file) == "ILSVRC2012_validation_ground_truth.txt") {
			std::string read = file_util::read_file_text(dir_or_file.c_str());
			read = file_util::to_local(read);
			image_classes = file_util::split_int64_sub_one(read, "\n");
		}
	}
	_class_count = 1000;
	if (image_classes.size() != image_files.size())
		return false;
	if (image_classes.size() == 0)
		return false;
	for (int64_t i = 0; i < image_files.size(); i++) 
		_image_class_files.push_back(std::pair<std::string,int64_t>(image_files[i], image_classes[i]));
	return true;
}
bool ADKDataset::init_dataset(const std::string& root){
	auto dirs_and_files = file_util::list_files_and_dirs(root);
	std::vector<std::string> image_files;
	std::vector<std::string> seg_files;
	for (auto dir_or_file : dirs_and_files) {
		if (file_util::is_dir(dir_or_file)) {
			auto file_name = file_util::get_file_name_without_ext(dir_or_file);
			bool is_image = file_name == "images";
			bool is_seg = file_name == "annotations";
			auto dirs_in = file_util::list_dirs(dir_or_file);
			for (auto dir_in : dirs_in){
				auto dir_in_name = file_util::get_file_name_without_ext(dir_in);
				if (_is_train&&dir_in_name=="training") {
					auto files = file_util::list_files(dir_in);
					for (auto file : files) {
						if (is_image)
							image_files.push_back(file);
						else if (is_seg)
							seg_files.push_back(file);
					}
				}
				else {
					auto files = file_util::list_files(dir_in);
					for (auto file : files) {
						if (is_image)
							image_files.push_back(file);
						else if (is_seg)
							seg_files.push_back(file);
					}
				}
			}
		}
	}
	if (seg_files.size() != image_files.size())
		return false;
	if ((image_files.size() == 0))
		return false;
	for (int64_t i = 0; i < image_files.size(); i++) {
		_image_seg_files.push_back(std::pair<std::string, std::string>(image_files[i], seg_files[i]));
	}
	_class_count = 150;
	return true;
}