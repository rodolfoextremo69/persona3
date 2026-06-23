#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_01_SHAPE_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_01_SHAPE_H

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace utec::tf {

    class Shape {
    private:
        std::vector<int> dims_;

        void validate() const {
            for (int dim : dims_) {
                if (dim <= 0) {
                    throw std::invalid_argument("toda dimension debe ser positiva");
                }
            }
        }

    public:
        Shape() = default;

        Shape(std::initializer_list<int> dims)
            : dims_(dims) {
            validate();
        }

        explicit Shape(std::vector<int> dims)
            : dims_(std::move(dims)) {
            validate();
        }

        std::size_t rank() const {
            return dims_.size();
        }

        std::size_t size() const {
            return dims_.size();
        }

        std::size_t numel() const {
            std::size_t total = 1;
            for (int dim : dims_) {
                total *= static_cast<std::size_t>(dim);
            }
            return total;
        }

        std::size_t total_size() const {
            return numel();
        }

        const std::vector<int>& dims() const {
            return dims_;
        }

        int operator[](std::size_t i) const {
            return dims_.at(i);
        }

        bool operator==(const Shape& other) const {
            return dims_ == other.dims_;
        }

        bool operator!=(const Shape& other) const {
            return !(*this == other);
        }
    };

}

#endif
