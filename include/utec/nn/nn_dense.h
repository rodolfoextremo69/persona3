#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_DENSE_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_DENSE_H

#include <stdexcept>
#include <unordered_map>
#include <string>
#include <memory>
#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_activation.h"
#include "utec/algebra/tensor_ops.h"

namespace utec::tf::layers {

class Dense : public Layer {
private:
    int units_;
    Activation activation_;
    Shape input_shape_;
    Shape output_shape_;
    Tensor<float> weights_;
    Tensor<float> bias_;
    // persona1
    Tensor<float> last_input_;
    Tensor<float> last_z_;
    Tensor<float> grad_weights_;
    Tensor<float> grad_bias_;
    // fin persona1
    bool built_ = false;

public:
    explicit Dense(int units, Activation activation = Activation::Linear)
        : units_(units), activation_(activation) {
        if (units <= 0) {
            throw std::invalid_argument("units invalidas");
        }
    }

    void build(const Shape& input_shape) override {
        if (input_shape.rank() != 1) {
            throw std::invalid_argument("Dense requiere input_shape rank 1");
        }

        input_shape_ = input_shape;
        output_shape_ = Shape{units_};

        const int in_features = input_shape[0];

        weights_ = Tensor<float>::ones(Shape{in_features, units_});
        bias_ = Tensor<float>::zeros(Shape{units_});

        for (int i = 0; i < in_features; ++i) {
            for (int j = 0; j < units_; ++j) {
                weights_(i, j) = 0.05f;
            }
        }

        built_ = true;
    }

    Tensor<float> forward(const Tensor<float>& x) override {
        if (!built_) {
            throw std::invalid_argument("Dense no construida");
        }

        if (x.rank() != 2) {
            throw std::invalid_argument("Dense requiere tensor rank 2");
        }

        if (x.shape()[1] != input_shape_[0]) {
            throw std::invalid_argument("shape incompatible en Dense");
        }

        last_input_ = x; // persona1

        Tensor<float> z = ops::matmul(x, weights_);

        for (int i = 0; i < z.shape()[0]; ++i) {
            for (int j = 0; j < z.shape()[1]; ++j) {
                z(i, j) += bias_(j);
            }
        }

        last_z_ = z; // persona1

        return apply_activation(z, activation_);
    }

    Shape output_shape() const override {
        return output_shape_;
    }

    std::unordered_map<std::string, Tensor<float>> parameters() const override {
        return {
            {"weights", weights_},
            {"bias", bias_}
        };
    }

    std::unordered_map<std::string, Tensor<float>> gradients() const override {
        return {
            {"weights", grad_weights_},
            {"bias", grad_bias_}
        };
    }

    // persona1
    Tensor<float> backward(const Tensor<float>& grad_output) override {
        Tensor<float> dZ = grad_output;

        if (activation_ == Activation::Relu) {
            Tensor<float> der = relu_derivative(last_z_);
            for (int i = 0; i < dZ.shape()[0]; ++i) {
                for (int j = 0; j < dZ.shape()[1]; ++j) {
                    dZ(i, j) *= der(i, j);
                }
            }
        }
        // Softmax + CategoricalCrossentropy: grad_output ya es y_pred - y_true

        grad_weights_ = ops::matmul(ops::transpose2d(last_input_), dZ);

        const int batch = dZ.shape()[0];
        const int units = dZ.shape()[1];

        grad_bias_ = Tensor<float>::zeros(Shape{units});
        for (int j = 0; j < units; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < batch; ++i) {
                sum += dZ(i, j);
            }
            grad_bias_(j) = sum;
        }

        return ops::matmul(dZ, ops::transpose2d(weights_));
    }

    [[nodiscard]] const Tensor<float>& grad_weights() const { return grad_weights_; }
    [[nodiscard]] const Tensor<float>& grad_bias() const { return grad_bias_; }
    [[nodiscard]] Tensor<float>& weights() { return weights_; }
    [[nodiscard]] Tensor<float>& bias() { return bias_; }
    // fin persona1

    std::unique_ptr<Layer> clone() const override {
        return std::make_unique<Dense>(*this);
    }

    std::string layer_type() const override {
        return "dense";
    }
};

}

#endif