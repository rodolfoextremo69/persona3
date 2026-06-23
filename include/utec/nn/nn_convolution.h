
#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_CONVOLUTION_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_CONVOLUTION_H

#include <array>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <memory>
#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_ops.h"

namespace utec::tf::layers {

class Conv2D : public Layer {
private:
    int filters_;
    std::array<int, 2> kernel_size_;
    Activation activation_;
    Shape input_shape_;
    Shape output_shape_;
    Tensor<float> weights_;
    Tensor<float> bias_;
    bool built_ = false;

public:
    Conv2D(int filters, std::array<int, 2> kernel_size,
           Activation activation = Activation::Linear)
        : filters_(filters), kernel_size_(kernel_size), activation_(activation) {
        if (filters <= 0) {
            throw std::invalid_argument("filters invalidos");
        }

        if (kernel_size[0] <= 0 || kernel_size[1] <= 0) {
            throw std::invalid_argument("kernel invalido");
        }
    }

    void build(const Shape& input_shape) override {
        if (input_shape.rank() != 3) {
            throw std::invalid_argument("Conv2D requiere input_shape rank 3");
        }

        const int h = input_shape[0];
        const int w = input_shape[1];
        const int c = input_shape[2];

        const int kh = kernel_size_[0];
        const int kw = kernel_size_[1];

        if (kh > h || kw > w) {
            throw std::invalid_argument("kernel mas grande que la entrada");
        }

        input_shape_ = input_shape;
        output_shape_ = Shape{h - kh + 1, w - kw + 1, filters_};

        weights_ = Tensor<float>::ones(Shape{kh, kw, c, filters_});
        bias_ = Tensor<float>::zeros(Shape{filters_});

        for (int i = 0; i < kh; ++i) {
            for (int j = 0; j < kw; ++j) {
                for (int ch = 0; ch < c; ++ch) {
                    for (int f = 0; f < filters_; ++f) {
                        weights_(i, j, ch, f) = 0.05f;
                    }
                }
            }
        }

        built_ = true;
    }

    Tensor<float> forward(const Tensor<float>& x) override {
        if (!built_) {
            throw std::invalid_argument("Conv2D no construida");
        }

        if (x.rank() != 4) {
            throw std::invalid_argument("Conv2D requiere tensor rank 4");
        }

        if (x.shape()[1] != input_shape_[0] ||
            x.shape()[2] != input_shape_[1] ||
            x.shape()[3] != input_shape_[2]) {
            throw std::invalid_argument("shape incompatible en Conv2D");
        }

        Tensor<float> out = ops::conv2d(x, weights_);

        for (int n = 0; n < out.shape()[0]; ++n) {
            for (int h = 0; h < out.shape()[1]; ++h) {
                for (int w = 0; w < out.shape()[2]; ++w) {
                    for (int f = 0; f < out.shape()[3]; ++f) {
                        out(n, h, w, f) += bias_(f);
                    }
                }
            }
        }

        return apply_activation(out, activation_);
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

    std::unique_ptr<Layer> clone() const override {
        return std::make_unique<Conv2D>(*this);
    }

    std::string layer_type() const override {
        return "conv2d";
    }
};

}

#endif
