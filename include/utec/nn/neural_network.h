#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NEURAL_NETWORK_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>

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
#include "utec/nn/nn_graph.h"

namespace utec::tf {

struct FitOptions {
    int epochs     = 1;
    int batch_size = 32;
};

struct History {
    std::vector<float> loss;
};

struct EvaluationResult {
    float loss = 0.0f;
};

class SequentialGraph {
public:
    void add(layers::Dense layer)        { layers_.push_back(layer.clone()); }
    void add(layers::Conv2D layer)       { layers_.push_back(layer.clone()); }
    void add(layers::MaxPooling2D layer) { layers_.push_back(layer.clone()); }
    void add(layers::Flatten layer)      { layers_.push_back(layer.clone()); }

    Tensor<float> forward(const Tensor<float>& x) {
        Tensor<float> out = x;
        for (auto& l : layers_) out = l->forward(out);
        return out;
    }

    Tensor<float> backward(const Tensor<float>& grad) {
        Tensor<float> g = grad;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
            g = (*it)->backward(g);
        return g;
    }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

class Sequential {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    bool compiled_ = false;
    optimizers::SGD optimizer_{0.01f};
    losses::CategoricalCrossentropy loss_{};
    Shape input_shape_;
    std::unordered_map<std::string, Tensor<float>> last_gradients_;

    Tensor<float> slice_batch(const Tensor<float>& t, int start, int end) const {
        int batch = end - start;
        const Shape& s = t.shape();

        std::vector<int> dims;
        dims.reserve(s.rank());
        dims.push_back(batch);
        for (std::size_t i = 1; i < s.rank(); ++i)
            dims.push_back(s[i]);

        Tensor<float> out = Tensor<float>::zeros(Shape(dims));
        std::size_t stride = t.numel() / static_cast<std::size_t>(s[0]);
        for (int i = 0; i < batch; ++i) {
            for (std::size_t j = 0; j < stride; ++j) {
                out[static_cast<std::size_t>(i) * stride + j] =
                    t[static_cast<std::size_t>(start + i) * stride + j];
            }
        }
        return out;
    }

    Tensor<float> initial_gradient(const Tensor<float>& y_pred,
                                    const Tensor<float>& y_true) const {
        if (!(y_pred.shape() == y_true.shape()))
            throw std::invalid_argument("shapes de y_pred y y_true incompatibles");
        float n = static_cast<float>(y_pred.shape()[0]);
        Tensor<float> g = Tensor<float>::zeros(y_pred.shape());
        for (std::size_t i = 0; i < y_pred.numel(); ++i)
            g[i] = (y_pred[i] - y_true[i]) / n;
        return g;
    }

    void validate_labels(const Tensor<float>& y_pred,
                         const Tensor<float>& y_true) const {
        if (y_pred.rank() != 2 || y_true.rank() != 2)
            throw std::invalid_argument("y_pred e y_true deben ser rank 2");
        if (y_pred.shape()[1] != y_true.shape()[1])
            throw std::invalid_argument(
                "etiquetas incompatibles: el modelo produce " +
                std::to_string(y_pred.shape()[1]) + " clases pero y tiene " +
                std::to_string(y_true.shape()[1]));
        if (y_pred.shape()[0] != y_true.shape()[0])
            throw std::invalid_argument("batch size de y_pred != y_true");
    }

    void apply_sgd() {
        for (auto& layer : layers_) {
            std::string type = layer->layer_type();

            if (type == "dense") {
                auto* dl = dynamic_cast<layers::Dense*>(layer.get());
                if (!dl) continue;
                auto grads = dl->gradients();
                auto& w = dl->weights();
                auto& b = dl->bias();
                if (grads.count("weights")) {
                    const auto& gw = grads.at("weights");
                    for (std::size_t i = 0; i < w.numel(); ++i)
                        w[i] -= optimizer_.learning_rate * gw[i];
                }
                if (grads.count("bias")) {
                    const auto& gb = grads.at("bias");
                    for (std::size_t i = 0; i < b.numel(); ++i)
                        b[i] -= optimizer_.learning_rate * gb[i];
                }

            } else if (type == "conv2d") {
                auto* cl = dynamic_cast<layers::Conv2D*>(layer.get());
                if (!cl) continue;
                auto grads  = cl->gradients();
                auto params = cl->parameters();
                if (grads.count("weights")) {
                    Tensor<float> w = params.at("weights");
                    const auto& gw  = grads.at("weights");
                    for (std::size_t i = 0; i < w.numel(); ++i)
                        w[i] -= optimizer_.learning_rate * gw[i];
                    cl->set_weights(w);
                }
                if (grads.count("bias")) {
                    Tensor<float> b = params.at("bias");
                    const auto& gb  = grads.at("bias");
                    for (std::size_t i = 0; i < b.numel(); ++i)
                        b[i] -= optimizer_.learning_rate * gb[i];
                    cl->set_bias(b);
                }
            }
        }
    }

    void collect_gradients() {
        last_gradients_.clear();
        int dense_count = 0;
        int conv_count  = 0;

        for (auto& layer : layers_) {
            auto grads = layer->gradients();
            if (grads.empty()) continue;

            std::string prefix;
            if (layer->layer_type() == "dense")
                prefix = "dense_" + std::to_string(dense_count++);
            else if (layer->layer_type() == "conv2d")
                prefix = "conv2d_" + std::to_string(conv_count++);
            else
                continue;

            for (const auto& item : grads)
                last_gradients_[prefix + "/" + item.first] = item.second;
        }
    }

public:
    Sequential() = default;

    template <typename LayerT>
    void add(const LayerT& layer) {
        std::unique_ptr<Layer> new_layer = layer.clone();

        if (layers_.empty()) {
            if (new_layer->layer_type() != "input")
                throw std::invalid_argument("la primera capa debe ser Input");
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
        if (layers_.empty())
            throw std::invalid_argument("modelo vacio");
        optimizer_ = optimizer;
        loss_      = loss;
        compiled_  = true;
    }

    bool compiled() const { return compiled_; }

    Tensor<float> predict(const Tensor<float>& batch) {
        if (layers_.empty()) throw std::invalid_argument("modelo vacio");
        if (batch.rank() != input_shape_.rank() + 1)
            throw std::invalid_argument("batch incompatible");
        for (std::size_t i = 0; i < input_shape_.rank(); ++i) {
            if (batch.shape()[i + 1] != input_shape_[i])
                throw std::invalid_argument("shape de batch incompatible");
        }
        Tensor<float> out = batch;
        for (auto& layer : layers_) out = layer->forward(out);
        return out;
    }

    EvaluationResult evaluate(const Tensor<float>& x, const Tensor<float>& y) {
        Tensor<float> y_pred = predict(x);
        validate_labels(y_pred, y);
        return EvaluationResult{loss_(y, y_pred)};
    }

    Tensor<float> backward(const Tensor<float>& loss_gradient) {
        if (!compiled_)
            throw std::logic_error("backward() requiere que el modelo este compilado.");
        Tensor<float> grad = loss_gradient;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
            grad = (*it)->backward(grad);
        collect_gradients();
        return grad;
    }

    Tensor<float> backward() {
        if (!compiled_)
            throw std::logic_error("backward() requiere que el modelo este compilado.");
        throw std::logic_error("backward() sin argumento requiere un gradiente inicial.");
    }

    History fit(const Tensor<float>& x, const Tensor<float>& y,
                const FitOptions& opts) {
        if (!compiled_)
            throw std::logic_error("fit() requiere que el modelo este compilado.");
        if (opts.epochs <= 0)
            throw std::invalid_argument("epochs debe ser > 0");
        if (opts.batch_size <= 0)
            throw std::invalid_argument("batch_size debe ser > 0");

        int n = static_cast<int>(x.shape()[0]);
        if (n == 0) throw std::invalid_argument("dataset vacio");

        {
            Tensor<float> y_pred_check = predict(x);
            validate_labels(y_pred_check, y);
        }

        History history;
        history.loss.reserve(static_cast<std::size_t>(opts.epochs));

        for (int epoch = 0; epoch < opts.epochs; ++epoch) {
            float epoch_loss = 0.0f;
            int   n_batches  = 0;

            for (int start = 0; start < n; start += opts.batch_size) {
                int end = std::min(start + opts.batch_size, n);

                Tensor<float> x_batch = slice_batch(x, start, end);
                Tensor<float> y_batch = slice_batch(y, start, end);

                Tensor<float> y_pred = predict(x_batch);

                epoch_loss += loss_(y_batch, y_pred);
                ++n_batches;

                Tensor<float> grad = initial_gradient(y_pred, y_batch);

                for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
                    grad = (*it)->backward(grad);

                collect_gradients();
                apply_sgd();
            }

            history.loss.push_back(epoch_loss / static_cast<float>(n_batches));
        }

        return history;
    }

    std::unordered_map<std::string, Tensor<float>> parameters() const {
        std::unordered_map<std::string, Tensor<float>> result;
        int dense_count = 0;
        int conv_count  = 0;

        for (const auto& layer : layers_) {
            auto params = layer->parameters();
            if (params.empty()) continue;

            std::string prefix;
            if (layer->layer_type() == "dense")
                prefix = "dense_" + std::to_string(dense_count++);
            else if (layer->layer_type() == "conv2d")
                prefix = "conv2d_" + std::to_string(conv_count++);
            else
                continue;

            for (const auto& item : params)
                result[prefix + "/" + item.first] = item.second;
        }
        return result;
    }

    std::unordered_map<std::string, Tensor<float>> last_gradients() const {
        return last_gradients_;
    }
};

}

#endif