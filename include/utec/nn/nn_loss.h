#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_LOSS_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_LOSS_H
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "utec/algebra/tensor_backend.h"

namespace utec::tf::losses {

    class CategoricalCrossentropy {
    public:
        float operator()(const Tensor<float>& y_true, const Tensor<float>& y_pred) const {
            if (!(y_true.shape() == y_pred.shape())) {
                throw std::invalid_argument("shapes incompatibles en CategoricalCrossentropy");
            }

            if (y_true.rank() != 2) {
                throw std::invalid_argument("CategoricalCrossentropy requiere tensores rank 2");
            }

            const int batch = y_true.shape()[0];
            const int classes = y_true.shape()[1];
            const float epsilon = 1e-7f;

            float total = 0.0f;

            for (int i = 0; i < batch; ++i) {
                for (int j = 0; j < classes; ++j) {
                    float pred = std::max(y_pred(i, j), epsilon);
                    total += -y_true(i, j) * std::log(pred);
                }
            }

            return total / static_cast<float>(batch);
        }
    };

}

#endif
