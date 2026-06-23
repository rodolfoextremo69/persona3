#ifndef UTEC_NN_OPTIMIZER_H
#define UTEC_NN_OPTIMIZER_H

#include <stdexcept>

namespace utec::tf::optimizers {

    struct SGD {
        float learning_rate;

        explicit SGD(float lr = 0.01f)
            : learning_rate(lr) {
            if (lr <= 0.0f) {
                throw std::invalid_argument("learning rate invalido");
            }
        }
    };

}

#endif
