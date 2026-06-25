#ifndef PROG3_PF_EPIC1_FEATURE2_V2026_1_UTEC_ALGEBRA_TENSOR_OPS_H
#define PROG3_PF_EPIC1_FEATURE2_V2026_1_UTEC_ALGEBRA_TENSOR_OPS_H

#include <stdexcept>
#include "utec/algebra/tensor_backend.h"

namespace utec::tf::ops {

template <typename T>
Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("multiply requiere tensores con el mismo shape");
    }

    Tensor<T> result = Tensor<T>::zeros(a.shape());

    if (a.rank() == 1) {
        for (int i = 0; i < a.shape()[0]; ++i) {
            result(i) = a(i) * b(i);
        }
    } else if (a.rank() == 2) {
        for (int i = 0; i < a.shape()[0]; ++i) {
            for (int j = 0; j < a.shape()[1]; ++j) {
                result(i, j) = a(i, j) * b(i, j);
            }
        }
    } else if (a.rank() == 3) {
        for (int i = 0; i < a.shape()[0]; ++i) {
            for (int j = 0; j < a.shape()[1]; ++j) {
                for (int k = 0; k < a.shape()[2]; ++k) {
                    result(i, j, k) = a(i, j, k) * b(i, j, k);
                }
            }
        }
    } else if (a.rank() == 4) {
        for (int n = 0; n < a.shape()[0]; ++n) {
            for (int h = 0; h < a.shape()[1]; ++h) {
                for (int w = 0; w < a.shape()[2]; ++w) {
                    for (int c = 0; c < a.shape()[3]; ++c) {
                        result(n, h, w, c) = a(n, h, w, c) * b(n, h, w, c);
                    }
                }
            }
        }
    } else {
        throw std::invalid_argument("multiply no soporta este rank");
    }

    return result;
}

template <typename T>
Tensor<T> transpose2d(const Tensor<T>& x) {
    if (x.rank() != 2) {
        throw std::invalid_argument("transpose2d requiere tensor de rank 2");
    }

    const int rows = x.shape()[0];
    const int cols = x.shape()[1];

    Tensor<T> result = Tensor<T>::zeros(Shape{cols, rows});

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(j, i) = x(i, j);
        }
    }

    return result;
}

template <typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.rank() != 2 || b.rank() != 2) {
        throw std::invalid_argument("matmul requiere tensores de rank 2");
    }

    const int m = a.shape()[0];
    const int k = a.shape()[1];
    const int k2 = b.shape()[0];
    const int n = b.shape()[1];

    if (k != k2) {
        throw std::invalid_argument("matmul recibio dimensiones incompatibles");
    }

    Tensor<T> result = Tensor<T>::zeros(Shape{m, n});

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T suma = T{};

            for (int t = 0; t < k; ++t) {
                suma += a(i, t) * b(t, j);
            }

            result(i, j) = suma;
        }
    }

    return result;
}

template <typename T>
Tensor<T> flatten_batch(const Tensor<T>& x) {
    if (x.rank() < 2) {
        throw std::invalid_argument("flatten_batch requiere rank >= 2");
    }

    const int batch = x.shape()[0];
    const int features = static_cast<int>(x.numel() / batch);

    return x.reshaped(Shape{batch, features});
}

template <typename T>
Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& kernel) {
    if (input.rank() != 4 || kernel.rank() != 4) {
        throw std::invalid_argument("conv2d requiere tensores de rank 4");
    }

    const int batch = input.shape()[0];
    const int in_h = input.shape()[1];
    const int in_w = input.shape()[2];
    const int in_c = input.shape()[3];

    const int k_h = kernel.shape()[0];
    const int k_w = kernel.shape()[1];
    const int k_c = kernel.shape()[2];
    const int filters = kernel.shape()[3];

    if (k_c != in_c) {
        throw std::invalid_argument("canales incompatibles en conv2d");
    }

    if (k_h <= 0 || k_w <= 0) {
        throw std::invalid_argument("kernel invalido en conv2d");
    }

    if (k_h > in_h || k_w > in_w) {
        throw std::invalid_argument("kernel mas grande que la entrada");
    }

    const int out_h = in_h - k_h + 1;
    const int out_w = in_w - k_w + 1;

    Tensor<T> output = Tensor<T>::zeros(Shape{batch, out_h, out_w, filters});

    for (int n = 0; n < batch; ++n) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                for (int f = 0; f < filters; ++f) {
                    T suma = T{};

                    for (int kh = 0; kh < k_h; ++kh) {
                        for (int kw = 0; kw < k_w; ++kw) {
                            for (int c = 0; c < in_c; ++c) {
                                suma += input(n, oh + kh, ow + kw, c) *
                                        kernel(kh, kw, c, f);
                            }
                        }
                    }

                    output(n, oh, ow, f) = suma;
                }
            }
        }
    }

    return output;
}

}  // namespace utec::tf::ops

namespace utec::tf::algebra {

template <typename T>
Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b) {
    return utec::tf::ops::multiply(a, b);
}

template <typename T>
Tensor<T> transpose2d(const Tensor<T>& x) {
    return utec::tf::ops::transpose2d(x);
}

template <typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    return utec::tf::ops::matmul(a, b);
}

template <typename T>
Tensor<T> flatten_batch(const Tensor<T>& x) {
    return utec::tf::ops::flatten_batch(x);
}

template <typename T>
Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& kernel) {
    return utec::tf::ops::conv2d(input, kernel);
}

}  // namespace utec::tf::algebra

#endif