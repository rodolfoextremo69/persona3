// persona1
#ifndef PROG3_PF_EPIC1_FEATURE4_NN_GRAPH_H
#define PROG3_PF_EPIC1_FEATURE4_NN_GRAPH_H

#include <string>
#include <unordered_map>
#include <cmath>
#include "utec/algebra/tensor_backend.h"

namespace utec::tf {

    using GradientMap = std::unordered_map<std::string, Tensor<float>>;

    class GraphContext {
    private:
        GradientMap gradients_;

    public:
        void clear_gradients() {
            gradients_.clear();
        }

        void set_gradient(const std::string& name, const Tensor<float>& gradient) {
            gradients_[name] = gradient;
        }

        [[nodiscard]] const GradientMap& last_gradients() const {
            return gradients_;
        }

        [[nodiscard]] bool empty() const {
            return gradients_.empty();
        }
    };

    inline bool allclose(const Tensor<float>& a, const Tensor<float>& b, float atol = 1e-6f) {
        if (a.shape() != b.shape()) {
            return false;
        }

        for (size_t i = 0; i < a.numel(); ++i) {
            if (std::fabs(a[i] - b[i]) > atol) {
                return false;
            }
        }

        return true;
    }

} // namespace utec::tf
// fin persona1

#endif
