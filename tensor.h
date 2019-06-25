#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
class tensor_load{
public:
	tensor_load() {}
	~tensor_load() {}
	bool static load_rgb(void* ptr,int64_t w,int64_t h,torch::Tensor &input_tensor);
	bool static load_rgba(void* ptr, int64_t w, int64_t h, torch::Tensor &input_tensor);
	bool static load(std::string file_name,cv::Size size,torch::Tensor &input_tensor,bool flip,int ro);
	bool static load_seg(std::string file_name, cv::Size size, torch::Tensor &input_tensor, bool flip,int ro);
};
template<typename Self>
class DatasetSegmentationBase :public torch::data::Dataset<Self> {
protected:
	std::vector<std::string> _zables;
	std::vector<std::pair<std::string, std::string>> _image_seg_files;
	cv::Size _size;
	bool _is_train;
	int64_t _class_count;
protected:
	void random_shuffle() {
		std::random_shuffle(_image_seg_files.begin(), _image_seg_files.end());
	}
public:
	explicit DatasetSegmentationBase(bool is_train,  int64_t w, int64_t h) {
		_is_train = is_train;
		_size = cv::Size(w, h);
	}
	virtual bool init_dataset(const std::string& root) = 0;
	std::shared_ptr<Self> split(double percent_eval=0.15) {
		random_shuffle();
		auto self = std::make_shared<Self>(false, _size.width, _size.height);
		auto all_size= _image_seg_files.size();
		auto size_eval = (int64_t)(all_size * percent_eval);
		auto size_train = all_size - size_eval;
		self->_image_seg_files = std::vector<std::pair<std::string, std::string>>(
			_image_seg_files.begin()+ size_train, _image_seg_files.end());
		self->_class_count = _class_count;
		_image_seg_files = std::vector<std::pair<std::string, std::string>>(
			_image_seg_files.begin(), _image_seg_files.begin()+ size_train);
		return self;
	}
	torch::data::Example<> get(size_t index)override {
		torch::data::Example<> example;
		auto file_name_image = _image_seg_files[index].first;
		auto file_name_seg = _image_seg_files[index].second;
		torch::Tensor image, seg;
		if (_is_train) {
			srand(GetTickCount());
			int f = rand() % 4;
			srand(1);
			int ro = 0;
			switch (f) {
			case 1:
				ro = 180;
				break;
			case 2:
				ro = 90;
				break;
			case 3:
				ro = 270;
				break;
			default:
				break;
			}
			if (!tensor_load::load(file_name_image, _size, image, f == 1, ro))
				return example;
			if (!tensor_load::load_seg(file_name_seg, _size, seg, f == 1, ro))
				return example;
		}
		else {
			if (!tensor_load::load(file_name_image, _size, image, false, 0))
				return example;
			if (!tensor_load::load_seg(file_name_seg, _size, seg, false, 0))
				return example;
		}
		example.data = image;		
		example.target = seg;	
		return example;
	}
	torch::optional<size_t> size() const override {
		return _image_seg_files.size();
	}
	int64_t get_class_count() {
		return _class_count+1;
	}
};
template<typename Self>
class DatasetClassBase :public  torch::data::Dataset<Self> {
protected:
	std::vector<std::pair<std::string,int64_t>> _image_class_files;
	std::set<std::string> _zables;
	cv::Size _size;
	bool _is_train;
	int64_t _class_count;
protected:
	void random_shuffle() {
		std::random_shuffle(_image_class_files.begin(), _image_class_files.end());
	}
public:
	explicit DatasetClassBase(bool is_train, int64_t w, int64_t h) {
		_is_train = is_train;
		_size = cv::Size(w, h);
	}
	virtual bool init_dataset(const std::string& root) = 0;
	std::shared_ptr<Self> split(double percent_eval = 0.15) {	
		random_shuffle();
		auto self = std::make_shared<Self>(false, _size.width, _size.height);
		auto all_size = _image_class_files.size();
		auto size_eval = (int64_t)(all_size * percent_eval);
		auto size_train = all_size - size_eval;
		self->_image_class_files = std::vector<std::pair<std::string, int64_t>>(
			_image_class_files.begin() + size_train, _image_class_files.end());
		self->_class_count = _class_count;
		_image_class_files = std::vector<std::pair<std::string,int64_t>>(
			_image_class_files.begin(), _image_class_files.begin() + size_train);
		return self;
	}
	torch::data::Example<> get(size_t index) override{
		torch::data::Example<> example;
		auto file_name_image = _image_class_files[index].first;
		auto file_class = _image_class_files[index].second;
		torch::Tensor image, seg;
		if (_is_train) {
			srand(GetTickCount());
			int f = rand() % 4;
			srand(1);
			int ro = 0;
			switch (f) {
			case 1:
				ro = 180;
				break;
			case 2:
				ro = 90;
				break;
			case 3:
				ro = 270;
				break;
			default:
				break;
			}
			if (!tensor_load::load(file_name_image, _size, image, f == 1, ro))
				return example;
			seg = torch::tensor(file_class, torch::kLong);
		}
		else {
			if (!tensor_load::load(file_name_image, _size, image, false, 0))
				return example;
			seg = torch::tensor(file_class, torch::kLong);
		}
		example.data = image;
		example.target = seg;
		return example;
	}
	torch::optional<size_t> size() const override {
		return _image_class_files.size();
	}
	virtual int64_t get_class_count() {
		return _class_count;
	}
};
class CustomDataset :public DatasetSegmentationBase<CustomDataset> {
public:
	explicit CustomDataset(bool is_train, int64_t w, int64_t h):
		DatasetSegmentationBase<CustomDataset>(is_train,w,h){}
	bool init_dataset(const std::string& root)override;
};
class CustomClassDataset :public DatasetClassBase<CustomClassDataset> {
public:
	explicit CustomClassDataset(bool is_train, int64_t w, int64_t h):
		DatasetClassBase<CustomClassDataset>(is_train, w, h) {}
	bool init_dataset(const std::string& root) override;
};
class ImageNetValDataset :public DatasetClassBase<ImageNetValDataset> {
public:
	explicit ImageNetValDataset(bool is_train, int64_t w, int64_t h) :
		DatasetClassBase<ImageNetValDataset>(is_train,w,h){}
	bool init_dataset(const std::string& root) override;
};
class ADKDataset :public DatasetSegmentationBase<ADKDataset> {
public:
	explicit ADKDataset(bool is_train, int64_t w, int64_t h):
		DatasetSegmentationBase<ADKDataset>(is_train, w, h){}
	bool init_dataset(const std::string& root) override;
};