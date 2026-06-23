#ifndef PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_ACTIVATION_H
#define PROG3_PF_EPIC1_FEATURE3_V2026_1_NN_ACTIVATION_H

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "utec/algebra/tensor_backend.h"

namespace utec::tf {

enum class Activation {
    Linear,
    Relu,
    Softmax
};

inline Tensor<float> apply_activation(const Tensor<float>& x, Activation activation) {
    Tensor<float> result = x;

    if (activation == Activation::Linear) {
        return result;
    }

    if (activation == Activation::Relu) {
        if (result.rank() == 1) {
            for (int i = 0; i < result.shape()[0]; ++i) {
                if (result(i) < 0.0f) result(i) = 0.0f;
            }
        } else if (result.rank() == 2) {
            for (int i = 0; i < result.shape()[0]; ++i) {
                for (int j = 0; j < result.shape()[1]; ++j) {
                    if (result(i, j) < 0.0f) result(i, j) = 0.0f;
                }
            }
        } else if (result.rank() == 4) {
            for (int n = 0; n < result.shape()[0]; ++n) {
                for (int h = 0; h < result.shape()[1]; ++h) {
                    for (int w = 0; w < result.shape()[2]; ++w) {
                        for (int c = 0; c < result.shape()[3]; ++c) {
                            if (result(n, h, w, c) < 0.0f) {
                                result(n, h, w, c) = 0.0f;
                            }
                        }
                    }
                }
            }
        } else {
            throw std::invalid_argument("Relu no soporta este rank");
        }

        return result;
    }

    if (activation == Activation::Softmax) {
        if (x.rank() != 2) {
            throw std::invalid_argument("Softmax requiere tensor rank 2");
        }

        const int batch = x.shape()[0];
        const int classes = x.shape()[1];

        for (int i = 0; i < batch; ++i) {
            float max_value = x(i, 0);

            for (int j = 1; j < classes; ++j) {
                max_value = std::max(max_value, x(i, j));
            }

            float sum = 0.0f;

            for (int j = 0; j < classes; ++j) {
                float value = std::exp(x(i, j) - max_value);
                result(i, j) = value;
                sum += value;
            }

            for (int j = 0; j < classes; ++j) {
                result(i, j) /= sum;
            }
        }

        return result;
    }

    return result;
}
    //persona1
inline Tensor<float> relu_derivative(const Tensor<float>& z) {
    Tensor<float> out = Tensor<float>::zeros(z.shape());

    if (z.rank() == 1) {
        for (int i = 0; i < z.shape()[0]; ++i) {
            out(i) = z(i) > 0.0f ? 1.0f : 0.0f;
        }
    } else if (z.rank() == 2) {
        for (int i = 0; i < z.shape()[0]; ++i) {
            for (int j = 0; j < z.shape()[1]; ++j) {
                out(i, j) = z(i, j) > 0.0f ? 1.0f : 0.0f;
            }
        }
    } else if (z.rank() == 4) {
        for (int n = 0; n < z.shape()[0]; ++n) {
            for (int h = 0; h < z.shape()[1]; ++h) {
                for (int w = 0; w < z.shape()[2]; ++w) {
                    for (int c = 0; c < z.shape()[3]; ++c) {
                        out(n, h, w, c) = z(n, h, w, c) > 0.0f ? 1.0f : 0.0f;
                    }
                }
            }
        }
    }

    return out;
}

inline Tensor<float> activation_derivative(const Tensor<float>& z, Activation activation) {
    if (activation == Activation::Relu) {
        return relu_derivative(z);
    }
    // Softmax + CategoricalCrossentropy: el gradiente ya llega como y_pred - y_true
    return Tensor<float>::ones(z.shape());
}
// fin persona1

}

#endif

