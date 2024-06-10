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
void Tensor<T>::print(std::ostream& os) const {
  os << "shape: ";
  for (auto i : shape_) {
    os << i << " ";
  }
  os << std::endl;

  os << "strides: ";
  for (auto i : strides_) {
    os << i << " ";
  }
  os << std::endl;

  os << "data: ";
  for (int64_t i = 0; i < size_; ++i) {
    os << data_.get()[i] << " ";
  }
  os << std::endl;
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

  T* new_data = new T[size];
  std::copy(data_.get(), data_.get() + size_, new_data);
  Tensor<T> tensor(shape, new_data);

  return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::view(const std::vector<int64_t>& shape) const {
  if (shape.empty()) {
    throw std::invalid_argument("shape must not be empty");
  }

  int64_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                 std::multiplies<int64_t>());
  if (size != size_) {
    throw std::invalid_argument("size mismatch");
  }

  Tensor<T> tensor(shape, data_);

  return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::transpose(int64_t axis1, int64_t axis2) const {
  if (axis1 >= shape_.size() || axis2 >= shape_.size()) {
    throw std::out_of_range("Dimension out of range");
  }

  std::vector<int64_t> new_shape = shape_;
  std::swap(new_shape[axis1], new_shape[axis2]);

  T* new_data = new T[size_];
  Tensor<T> new_tensor(new_shape, new_data);
  std::vector<int64_t> new_strides = new_tensor.strides();

  auto compute_offset = [](const std::vector<int64_t>& indices,
                           const std::vector<int64_t>& strides) -> int64_t {
    int64_t offset = 0;
    for (int64_t i = 0; i < indices.size(); ++i) {
      offset += indices[i] * strides[i];
    }
    return offset;
  };

  std::vector<int64_t> old_indices(shape_.size(), 0);
  std::vector<int64_t> new_indices(new_shape.size(), 0);
  int64_t old_index = 0;
  int64_t new_index = 0;
  for (int64_t i = 0; i < size_; ++i) {
    old_index = compute_offset(old_indices, strides_);
    new_indices = old_indices;
    std::swap(new_indices[axis1], new_indices[axis2]);
    new_index = compute_offset(new_indices, new_strides);
    new_tensor[new_index] = (*this)[old_index];
    for (int64_t j = shape_.size() - 1; j >= 0; --j) {
      if (old_indices[j] + 1 < shape_[j]) {
        ++old_indices[j];
        break;
      } else {
        old_indices[j] = 0;
      }
    }
  }
  return new_tensor;
}

template <typename T>
Tensor<T> Tensor<T>::clone() const {
  T* new_data = new T[size_];
  std::copy(data_.get(), data_.get() + size_, new_data);
  Tensor<T> tensor(shape_, new_data);
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