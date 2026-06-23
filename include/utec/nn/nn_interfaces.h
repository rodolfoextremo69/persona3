#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_INTERFACES_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_INTERFACES_H

#include <memory>
#include <unordered_map>
#include <string>

#include "utec/algebra/shape.h"
#include "utec/algebra/tensor_backend.h"

namespace utec::tf {

    class Layer {
    public:
        virtual ~Layer() = default;

        virtual void build(const Shape& input_shape) = 0;
        virtual Tensor<float> forward(const Tensor<float>& x) = 0;
        // persona1
        virtual Tensor<float> backward(const Tensor<float>& grad_output) {
            return grad_output;
        }
        // fin persona1
        virtual Shape output_shape() const = 0;

        virtual std::unordered_map<std::string, Tensor<float>> parameters() const {
            return {};
        }

        virtual std::unique_ptr<Layer> clone() const = 0;

        virtual std::string layer_type() const {
            return "layer";
        }
    };

    namespace layers {
        class Input : public Layer {
        private:
            Shape shape_;

        public:
            explicit Input(const Shape& shape)
                : shape_(shape) {}

            void build(const Shape& input_shape) override {
                (void)input_shape;
            }

            Tensor<float> forward(const Tensor<float>& x) override {
                return x;
            }

            Shape output_shape() const override {
                return shape_;
            }

            std::unordered_map<std::string, Tensor<float>> parameters() const override {
                return {};
            }

            std::unique_ptr<Layer> clone() const override {
                return std::make_unique<Input>(*this);
            }

            std::string layer_type() const override {
                return "input";
            }
        };
    }

}

#endif
