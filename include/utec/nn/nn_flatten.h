#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_FLATTEN_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_FLATTEN_H


#include <stdexcept>
#include <memory>
#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_ops.h"

namespace utec::tf::layers {

    class Flatten : public Layer {
    private:
        Shape input_shape_;
        Shape output_shape_;
        bool built_ = false;
        // persona2
        Shape last_input_batch_shape_;
        // fin persona2

    public:
        Flatten() = default;

        void build(const Shape& input_shape) override {
            if (input_shape.rank() < 1) {
                throw std::invalid_argument("Flatten requiere input_shape valido");
            }

            input_shape_ = input_shape;
            output_shape_ = Shape{static_cast<int>(input_shape.numel())};
            built_ = true;
        }

        Tensor<float> forward(const Tensor<float>& x) override {
            if (!built_) {
                throw std::invalid_argument("Flatten no construida");
            }

            if (x.rank() < 2) {
                throw std::invalid_argument("Flatten requiere tensor con batch");
            }

            for (std::size_t i = 0; i < input_shape_.rank(); ++i) {
                if (x.shape()[i + 1] != input_shape_[i]) {
                    throw std::invalid_argument("shape incompatible en Flatten");
                }
            }
            // persona2
            last_input_batch_shape_ = x.shape();
            // fin persona2

            return ops::flatten_batch(x);
        }
        // persona2
        Tensor<float> backward(const Tensor<float>& grad_output) override {
            return grad_output.reshaped(last_input_batch_shape_);
        }
        // fin persona2

        Shape output_shape() const override {
            return output_shape_;
        }

        std::unique_ptr<Layer> clone() const override {
            return std::make_unique<Flatten>(*this);
        }

        std::string layer_type() const override {
            return "flatten";
        }
    };

}


#endif
