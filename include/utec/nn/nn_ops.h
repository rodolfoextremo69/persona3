
#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_OPS_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_OPS_H

#include <stdexcept>
#include "utec/algebra/tensor_backend.h"
#include "utec/algebra/tensor_ops.h"

namespace utec::tf {

    enum class Padding {
        Valid,
        Same
    };

    struct Strides {
        int h;
        int w;

        Strides(int h_value = 1, int w_value = 1)
            : h(h_value), w(w_value) {
            if (h <= 0 || w <= 0) {
                throw std::invalid_argument("strides invalidos");
            }
        }
    };

}

#endif