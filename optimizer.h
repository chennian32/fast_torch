#pragma once
#include <torch/torch.h>
#include <cstddef>
#include <utility>
#include <vector>
#include "model.h"
struct SGDOptions {
	/* implicit */ SGDOptions::SGDOptions(double learning_rate) : learning_rate_(learning_rate) {}
	TORCH_ARG(double, learning_rate);
	TORCH_ARG(double, momentum) = 0;
	TORCH_ARG(double, dampening) = 0;
	TORCH_ARG(double, weight_decay) = 0;
	TORCH_ARG(bool, nesterov) = false;
};

class SGD : public torch::optim::Optimizer {
public:
	template <typename ParameterContainer>
	explicit SGD(ParameterContainer&& parameters, const SGDOptions& options)
		: Optimizer(std::forward<ParameterContainer>(parameters)),
		options(options) {}
	torch::Tensor& at_buffer(size_t index) {
		if (momentum_buffers.size() <= index) {
			momentum_buffers.reserve(index);
			for (auto i = momentum_buffers.size(); i <= index; ++i) {
				momentum_buffers.push_back(torch::zeros_like(parameters_.at(i)));
			}
		}
		const auto& parameter = parameters_.at(index);
		const auto& buffer = momentum_buffers.at(index);
		if (buffer.device() != parameter.device() ||
			buffer.dtype() != parameter.dtype()) {
			momentum_buffers[index] = buffer.to(parameter.device(), parameter.scalar_type());
		}
		return momentum_buffers[index];
	}
	void step() override {
		using namespace torch;
		for (size_t i = 0; i < parameters_.size(); ++i) {
			Tensor p = parameters_.at(i);

			if (!p.grad().defined()) {
				continue;
			}

			auto update = p.grad();

			if (options.weight_decay_ > 0) {
				update += options.weight_decay_ * p;
			}
			if (options.momentum_ != 0) {
				const auto dampening = iteration_ == 0 ? 1 : 1 - options.dampening_;
				auto& momentum = at_buffer(i);
				momentum = (options.momentum_ * momentum) + (dampening * update);
				if (options.nesterov_) {
					update += options.momentum_ * momentum;
				}
				else {
					update = momentum;
				}
				momentum_buffers[i]=momentum.detach_();
			}

			NoGradGuard guard;
			p.add_(-options.learning_rate_ * update);
		}
		iteration_ += 1;
	}

	void save(torch::serialize::OutputArchive& archive) const override{
		torch::optim::serialize(archive, "momentum_buffers", momentum_buffers);
	}
	void load(torch::serialize::InputArchive& archive) override {
		torch::optim::serialize(archive, "momentum_buffers", momentum_buffers);
	}

	SGDOptions options;

	std::vector<torch::Tensor> momentum_buffers;

private:
	SGD() : options(0) {}

	/// Counts how often `step()` is called, for dampening.
	size_t iteration_{ 0 };
};