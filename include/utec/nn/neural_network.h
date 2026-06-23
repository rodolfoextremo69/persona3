#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NEURAL_NETWORK_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <stdexcept>

#include "utec/algebra/shape.h"
#include "utec/algebra/tensor_backend.h"

#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"
#include "utec/nn/nn_ops.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_convolution.h"
#include "utec/nn/nn_pooling.h"
#include "utec/nn/nn_flatten.h"

namespace utec::tf {

class Sequential {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    bool compiled_ = false;
    optimizers::SGD optimizer_{0.01f};
    losses::CategoricalCrossentropy loss_{};
    Shape input_shape_;

public:
    Sequential() = default;

    template <typename LayerT>
    void add(const LayerT& layer) {
        std::unique_ptr<Layer> new_layer = layer.clone();

        if (layers_.empty()) {
            if (new_layer->layer_type() != "input") {
                throw std::invalid_argument("la primera capa debe ser Input");
            }

            Shape initial_shape = new_layer->output_shape();
            new_layer->build(initial_shape);
            input_shape_ = initial_shape;
        } else {
            Shape previous_shape = layers_.back()->output_shape();
            new_layer->build(previous_shape);
        }

        layers_.push_back(std::move(new_layer));
    }

    void compile(optimizers::SGD optimizer, losses::CategoricalCrossentropy loss) {
        if (layers_.empty()) {
            throw std::invalid_argument("modelo vacio");
        }

        optimizer_ = optimizer;
        loss_ = loss;
        compiled_ = true;
    }

    bool compiled() const {
        return compiled_;
    }

    Tensor<float> predict(const Tensor<float>& batch) {
        if (layers_.empty()) {
            throw std::invalid_argument("modelo vacio");
        }

        if (batch.rank() != input_shape_.rank() + 1) {
            throw std::invalid_argument("batch incompatible");
        }

        for (std::size_t i = 0; i < input_shape_.rank(); ++i) {
            if (batch.shape()[i + 1] != input_shape_[i]) {
                throw std::invalid_argument("shape de batch incompatible");
            }
        }

        Tensor<float> out = batch;

        for (auto& layer : layers_) {
            out = layer->forward(out);
        }

        return out;
    }

    std::unordered_map<std::string, Tensor<float>> parameters() const {
        std::unordered_map<std::string, Tensor<float>> result;

        int dense_count = 0;
        int conv_count = 0;

        for (const auto& layer : layers_) {
            auto params = layer->parameters();

            if (params.empty()) {
                continue;
            }

            std::string prefix;

            if (layer->layer_type() == "dense") {
                prefix = "dense_" + std::to_string(dense_count++);
            } else if (layer->layer_type() == "conv2d") {
                prefix = "conv2d_" + std::to_string(conv_count++);
            } else {
                continue;
            }

            for (const auto& item : params) {
                result[prefix + "/" + item.first] = item.second;
            }
        }

        return result;
    }
};

}

#endif
