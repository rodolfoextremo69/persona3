#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_POOLING_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_POOLING_H

#include <array>
#include <stdexcept>
#include <memory>
#include <unordered_map>
#include <string>
#include <algorithm>
#include "utec/nn/nn_interfaces.h"

namespace utec::tf::layers {

class MaxPooling2D : public Layer {
private:
    std::array<int, 2> pool_size_;
    std::array<int, 2> strides_;
    Shape input_shape_;
    Shape output_shape_;
    bool built_ = false;
    // persona2
    Tensor<float> max_mask_;
    // fin persona2

public:
    explicit MaxPooling2D(std::array<int, 2> pool_size)
        : pool_size_(pool_size), strides_(pool_size) {
        if (pool_size_[0] <= 0 || pool_size_[1] <= 0) {
            throw std::invalid_argument("ventana invalida en MaxPooling2D");
        }
    }

    void build(const Shape& input_shape) override {
        if (input_shape.rank() != 3) {
            throw std::invalid_argument("MaxPooling2D requiere input_shape rank 3");
        }

        const int h = input_shape[0];
        const int w = input_shape[1];
        const int c = input_shape[2];

        const int pool_h = pool_size_[0];
        const int pool_w = pool_size_[1];

        if (pool_h > h || pool_w > w) {
            throw std::invalid_argument("ventana mas grande que la entrada");
        }

        if (h % pool_h != 0 || w % pool_w != 0) {
            throw std::invalid_argument("la ventana debe dividir exactamente la entrada");
        }

        input_shape_ = input_shape;
        output_shape_ = Shape{h / pool_h, w / pool_w, c};
        built_ = true;
    }

    Tensor<float> forward(const Tensor<float>& x) override {
        if (!built_) {
            throw std::invalid_argument("MaxPooling2D no construida");
        }

        if (x.rank() != 4) {
            throw std::invalid_argument("MaxPooling2D requiere tensor rank 4");
        }

        if (x.shape()[1] != input_shape_[0] ||
            x.shape()[2] != input_shape_[1] ||
            x.shape()[3] != input_shape_[2]) {
            throw std::invalid_argument("shape incompatible en MaxPooling2D");
        }

        const int batch = x.shape()[0];
        const int channels = x.shape()[3];

        const int pool_h = pool_size_[0];
        const int pool_w = pool_size_[1];

        const int stride_h = strides_[0];
        const int stride_w = strides_[1];

        const int out_h = output_shape_[0];
        const int out_w = output_shape_[1];

        Tensor<float> out = Tensor<float>::zeros(Shape{batch, out_h, out_w, channels});
        // persona2
        max_mask_ = Tensor<float>::zeros(x.shape());
        // fin persona2

        for (int n = 0; n < batch; ++n) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    for (int c = 0; c < channels; ++c) {

                        // persona2
                        int best_h = oh * stride_h;
                        int best_w = ow * stride_w;

                        float best = x(n, best_h, best_w, c);

                        for (int kh = 0; kh < pool_h; ++kh) {
                            for (int kw = 0; kw < pool_w; ++kw) {

                                const int ih = oh * stride_h + kh;
                                const int iw = ow * stride_w + kw;

                                if (x(n, ih, iw, c) > best) {
                                    best = x(n, ih, iw, c);
                                    best_h = ih;
                                    best_w = iw;
                                }
                            }
                        }

                        max_mask_(n, best_h, best_w, c) = 1.0f;

                        out(n, oh, ow, c) = best;
                        // fin persona2
                    }
                }
            }
        }

        return out;
    }
                    // persona2
                    Tensor<float> backward(const Tensor<float>& grad_output) override {

                        const int batch = grad_output.shape()[0];
                        const int channels = grad_output.shape()[3];

                        const int pool_h = pool_size_[0];
                        const int pool_w = pool_size_[1];

                        const int stride_h = strides_[0];
                        const int stride_w = strides_[1];

                        const int out_h = output_shape_[0];
                        const int out_w = output_shape_[1];

                        Tensor<float> grad_input =
                            Tensor<float>::zeros(
                                Shape{
                                    batch,
                                    input_shape_[0],
                                    input_shape_[1],
                                    input_shape_[2]
                                });

                        for (int n = 0; n < batch; ++n) {
                            for (int oh = 0; oh < out_h; ++oh) {
                                for (int ow = 0; ow < out_w; ++ow) {
                                    for (int c = 0; c < channels; ++c) {

                                        float grad = grad_output(n, oh, ow, c);

                                        for (int kh = 0; kh < pool_h; ++kh) {
                                            for (int kw = 0; kw < pool_w; ++kw) {

                                                int ih = oh * stride_h + kh;
                                                int iw = ow * stride_w + kw;

                                                if (max_mask_(n, ih, iw, c) > 0.5f) {
                                                    grad_input(n, ih, iw, c) += grad;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        return grad_input;
                    }
                    // fin persona2
    Shape output_shape() const override {
        return output_shape_;
    }

    std::unordered_map<std::string, Tensor<float>> parameters() const override {
        return {};
    }


    std::unordered_map<std::string, Tensor<float>> gradients() const override {
        return {};
    }

    std::unique_ptr<Layer> clone() const override {
        return std::make_unique<MaxPooling2D>(*this);
    }

    std::string layer_type() const override {
        return "maxpooling2d";
    }
};

}

#endif