#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_01_TENSOR_BACKEND_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_01_TENSOR_BACKEND_H

#include <Eigen/Dense>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "utec/algebra/shape.h"

namespace utec::tf {

template <typename T>
class Tensor {
private:
    Shape shape_;
    Eigen::Array<T, Eigen::Dynamic, 1> data_;

    std::size_t flat_index(std::span<const int> indices) const {
        if (indices.size() != shape_.rank()) {
            throw std::invalid_argument("numero de indices incompatible con el rank");
        }

        std::size_t flat = 0;
        std::size_t stride = 1;

        for (std::size_t i = shape_.rank(); i-- > 0;) {
            int dim = shape_[i];
            int idx = indices[i];

            if (idx < 0 || idx >= dim) {
                throw std::out_of_range("indice fuera de rango");
            }

            flat += static_cast<std::size_t>(idx) * stride;
            stride *= static_cast<std::size_t>(dim);
        }

        return flat;
    }

public:
    Tensor() = default;

    explicit Tensor(const Shape& shape)
        : shape_(shape), data_(static_cast<Eigen::Index>(shape.numel())) {
        data_.setZero();
    }

    Tensor(const Shape& shape, const T& value)
        : shape_(shape), data_(static_cast<Eigen::Index>(shape.numel())) {
        data_.setConstant(value);
    }

    Tensor(const Shape& shape, const std::vector<T>& values)
        : shape_(shape), data_(static_cast<Eigen::Index>(shape.numel())) {
        if (values.size() != shape.numel()) {
            throw std::invalid_argument("cantidad de datos incompatible con la forma");
        }

        for (std::size_t i = 0; i < values.size(); ++i) {
            data_(static_cast<Eigen::Index>(i)) = values[i];
        }
    }

    static Tensor<T> zeros(const Shape& shape) {
        Tensor<T> out(shape);
        out.data_.setZero();
        return out;
    }

    static Tensor<T> ones(const Shape& shape) {
        Tensor<T> out(shape);
        out.data_.setOnes();
        return out;
    }

    static Tensor<T> from_data(const Shape& shape, const std::vector<T>& values) {
        return Tensor<T>(shape, values);
    }

    const Shape& shape() const {
        return shape_;
    }

    std::size_t rank() const {
        return shape_.rank();
    }

    std::size_t numel() const {
        return shape_.numel();
    }

    std::size_t size() const {
        return numel();
    }

    T& operator[](std::size_t index) {
        if (index >= numel()) {
            throw std::out_of_range("indice fuera de rango");
        }

        return data_(static_cast<Eigen::Index>(index));
    }

    const T& operator[](std::size_t index) const {
        if (index >= numel()) {
            throw std::out_of_range("indice fuera de rango");
        }

        return data_(static_cast<Eigen::Index>(index));
    }

    T& at(std::initializer_list<int> indices) {
        std::vector<int> idx(indices);

        return data_(static_cast<Eigen::Index>(
            flat_index(std::span<const int>(idx.data(), idx.size()))
        ));
    }

    const T& at(std::initializer_list<int> indices) const {
        std::vector<int> idx(indices);

        return data_(static_cast<Eigen::Index>(
            flat_index(std::span<const int>(idx.data(), idx.size()))
        ));
    }

    template <typename... Ix>
    T& operator()(Ix... indices) {
        static_assert((std::is_integral_v<Ix> && ...), "todos los indices deben ser enteros");

        const int idx[] = {static_cast<int>(indices)...};

        return data_(static_cast<Eigen::Index>(
            flat_index(std::span<const int>(idx, sizeof...(indices)))
        ));
    }

    template <typename... Ix>
    const T& operator()(Ix... indices) const {
        static_assert((std::is_integral_v<Ix> && ...), "todos los indices deben ser enteros");

        const int idx[] = {static_cast<int>(indices)...};

        return data_(static_cast<Eigen::Index>(
            flat_index(std::span<const int>(idx, sizeof...(indices)))
        ));
    }

    Tensor<T>& reshape(const Shape& new_shape) {
        if (new_shape.numel() != numel()) {
            throw std::invalid_argument("reshape incompatible");
        }

        shape_ = new_shape;
        return *this;
    }

    Tensor<T> reshaped(const Shape& new_shape) const {
        if (new_shape.numel() != numel()) {
            throw std::invalid_argument("reshape incompatible");
        }

        Tensor<T> out(new_shape);
        out.data_ = data_;
        return out;
    }

    Tensor<T> operator+(const Tensor<T>& other) const {
        if (!(shape_ == other.shape_)) {
            throw std::invalid_argument("shapes incompatibles para suma");
        }

        Tensor<T> out(shape_);
        out.data_ = data_ + other.data_;
        return out;
    }

    Tensor<T> operator-(const Tensor<T>& other) const {
        if (!(shape_ == other.shape_)) {
            throw std::invalid_argument("shapes incompatibles para resta");
        }

        Tensor<T> out(shape_);
        out.data_ = data_ - other.data_;
        return out;
    }
};

}

#endif
