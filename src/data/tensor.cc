#include "data/tensor.h"

#include <glog/logging.h>

#include <algorithm>
#include <stdexcept>

namespace engine {
namespace tensor {

template <typename T>
Tensor<T>::Tensor(const std::vector<int64_t>& shape,
                  const std::shared_ptr<T>& data)
    : shape_(shape), strides_(shape.size()), data_(data) {
  if (shape.empty()) {
    throw std::invalid_argument("shape must not be empty");
  }
  if (data == nullptr) {
    throw std::invalid_argument("data must not be null");
  }

  size_ = std::accumulate(shape.begin(), shape.end(), 1,
                          std::multiplies<int64_t>());
  strides_[shape.size() - 1] = 1;
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides_[i] = shape_[i + 1] * strides_[i + 1];
  }
}

template <typename T>
Tensor<T>::Tensor() : size_(0), shape_(), strides_(), data_(nullptr) {}

template <typename T>
Tensor<T>::Tensor(const std::vector<int64_t>& shape)
    : shape_(shape), strides_(shape.size()) {
  if (shape.empty()) {
    throw std::invalid_argument("shape must not be empty");
  }

  size_ = std::accumulate(shape.begin(), shape.end(), 1,
                          std::multiplies<int64_t>());

  data_ = std::shared_ptr<T>(new T[size_], std::default_delete<T[]>());
  std::fill(data_.get(), data_.get() + size_, static_cast<T>(0));

  strides_[shape.size() - 1] = 1;
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides_[i] = shape_[i + 1] * strides_[i + 1];
  }
}

template <typename T>
Tensor<T>::Tensor(const std::vector<int64_t>& shape, const T* data)
    : shape_(shape), strides_(shape.size()) {
  if (shape.empty()) {
    throw std::invalid_argument("shape must not be empty");
  }
  if (data == nullptr) {
    throw std::invalid_argument("data must not be null");
  }

  size_ = std::accumulate(shape.begin(), shape.end(), 1,
                          std::multiplies<int64_t>());
  data_ = std::shared_ptr<T>(new T[size_], std::default_delete<T[]>());
  std::copy(data, data + size_, data_.get());

  strides_[shape.size() - 1] = 1;
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides_[i] = shape_[i + 1] * strides_[i + 1];
  }
}

template <typename T>
Tensor<T>::Tensor(const std::vector<int64_t>& shape, const std::vector<T>& data)
    : shape_(shape), strides_(shape.size()) {
  if (shape.empty()) {
    throw std::invalid_argument("shape must not be empty");
  }
  if (data.empty()) {
    throw std::invalid_argument("data must not be empty");
  }

  size_ = std::accumulate(shape.begin(), shape.end(), 1,
                          std::multiplies<int64_t>());
  data_ = std::shared_ptr<T>(new T[size_], std::default_delete<T[]>());
  std::copy(data.begin(), data.end(), data_.get());

  strides_[shape.size() - 1] = 1;
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides_[i] = shape_[i + 1] * strides_[i + 1];
  }
}

template <typename T>
Tensor<T>::Tensor(const Tensor& other)
    : size_(other.size_),
      shape_(other.shape_),
      strides_(other.strides_),
      data_(other.data_) {}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
  if (this != &other) {
    size_ = other.size_;
    shape_ = other.shape_;
    strides_ = other.strides_;
    data_ = other.data_;
  }
  return *this;
}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : size_(std::exchange(other.size_, 0)),
      shape_(std::exchange(other.shape_, {})),
      strides_(std::exchange(other.strides_, {})),
      data_(std::exchange(other.data_, nullptr)) {}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    size_ = std::exchange(other.size_, 0);
    shape_ = std::exchange(other.shape_, {});
    strides_ = std::exchange(other.strides_, {});
    data_ = std::exchange(other.data_, nullptr);
  }
  return *this;
}

template <typename T>
Tensor<T>::~Tensor() {}

template <typename T>
T& Tensor<T>::operator[](int64_t index) {
  if (index < 0 || index >= size_) {
    throw std::out_of_range("index out of range");
  }
  return data_.get()[index];
}

template <typename T>
const T& Tensor<T>::operator[](int64_t index) const {
  if (index < 0 || index >= size_) {
    throw std::out_of_range("index out of range");
  }
  return data_.get()[index];
}

template <typename T>
T& Tensor<T>::operator[](const std::vector<int64_t>& indices) {
  int64_t index = 0;
  for (int64_t i = 0; i < indices.size(); ++i) {
    index += indices[i] * strides_[i];
  }
  if (index < 0 || index >= size_) {
    throw std::out_of_range("index out of range");
  }
  return data_.get()[index];
}

template <typename T>
const T& Tensor<T>::operator[](const std::vector<int64_t>& indices) const {
  int64_t index = 0;
  for (int64_t i = 0; i < indices.size(); ++i) {
    index += indices[i] * strides_[i];
  }
  if (index < 0 || index >= size_) {
    throw std::out_of_range("index out of range");
  }
  return data_.get()[index];
}

template <typename T>
T& Tensor<T>::at(int64_t index) {
  return (*this)[index];
}

template <typename T>
const T& Tensor<T>::at(int64_t index) const {
  return (*this)[index];
}

template <typename T>
T& Tensor<T>::at(const std::vector<int64_t>& indices) {
  return (*this)[indices];
}

template <typename T>
const T& Tensor<T>::at(const std::vector<int64_t>& indices) const {
  return (*this)[indices];
}

template <typename T>
T* Tensor<T>::data() {
  return data_.get();
}

template <typename T>
const T* Tensor<T>::data() const {
  return data_.get();
}

template <typename T>
size_t Tensor<T>::size() const {
  return size_;
}

template <typename T>
const std::vector<int64_t>& Tensor<T>::shape() const {
  return shape_;
}

template <typename T>
const std::vector<int64_t>& Tensor<T>::strides() const {
  return strides_;
}

template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<int64_t>& shape) const {
  if (shape.empty()) {
    throw std::invalid_argument("shape must not be empty");
  }

  int64_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                 std::multiplies<int64_t>());
  if (size != size_) {
    throw std::invalid_argument("size mismatch");
  }

  Tensor<T> tensor;
  tensor.size_ = size;
  tensor.shape_ = shape;
  tensor.strides_ = shape;
  tensor.data_ = data_;

  tensor.strides_[shape.size() - 1] = 1;
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    tensor.strides_[i] = shape[i + 1] * tensor.strides_[i + 1];
  }

  return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::transpose(const std::vector<int64_t>& axes) const {
  if (axes.empty()) {
    throw std::invalid_argument("axes must not be empty");
  }
  if (axes.size() != shape_.size()) {
    throw std::invalid_argument("axes size mismatch");
  }

  std::vector<int64_t> shape(shape_.size());
  std::vector<int64_t> strides(strides_.size());
  for (int64_t i = 0; i < axes.size(); ++i) {
    shape[i] = shape_[axes[i]];
    strides[i] = strides_[axes[i]];
  }

  Tensor<T> tensor;
  tensor.size_ = size_;
  tensor.shape_ = shape;
  tensor.strides_ = strides;
  tensor.data_ = data_;

  return tensor;
}

template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int8_t>;
template class Tensor<int16_t>;
template class Tensor<int32_t>;
template class Tensor<int64_t>;

}  // namespace tensor
}  // namespace engine