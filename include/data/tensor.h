#pragma once
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <vector>

namespace engine {
namespace tensor {
template <typename T>
class Tensor {
 private:
  explicit Tensor(const std::vector<int64_t>& shape,
                  const std::shared_ptr<T>& data);

 public:
  explicit Tensor();

  explicit Tensor(const std::vector<int64_t>& shape);

  explicit Tensor(const std::vector<int64_t>& shape, const T* data);

  explicit Tensor(const std::vector<int64_t>& shape,
                  const std::vector<T>& data);

  Tensor(const Tensor& other);

  Tensor& operator=(const Tensor& other);

  Tensor(Tensor&& other) noexcept;

  Tensor& operator=(Tensor&& other) noexcept;

  ~Tensor();

  void print(std::ostream& os) const;

  T& operator[](int64_t index);

  const T& operator[](int64_t index) const;

  T& operator[](const std::vector<int64_t>& indices);

  const T& operator[](const std::vector<int64_t>& indices) const;

  T& at(int64_t index);

  const T& at(int64_t index) const;

  T& at(const std::vector<int64_t>& indices);

  const T& at(const std::vector<int64_t>& indices) const;

  T* data();

  const T* data() const;

  size_t size() const;

  const std::vector<int64_t>& shape() const;

  const std::vector<int64_t>& strides() const;

  Tensor reshape(const std::vector<int64_t>& shape) const;

  Tensor transpose(int64_t axis1, int64_t axis2) const;

  Tensor squeeze() const;

  Tensor squeeze(int64_t axis) const;

  Tensor unsqueeze() const;

  Tensor unsqueeze(int64_t axis) const;

  Tensor view(const std::vector<int64_t>& shape) const;

  Tensor expand(const std::vector<int64_t>& shape) const;

  Tensor clone() const;

  Tensor contiguous() const;

  Tensor to(const std::string& device) const;

  Tensor to(const std::string& device, int64_t dtype) const;

  Tensor to(int64_t dtype) const;

  void transform(Tensor& other, const std::function<T(T)>& f) const;

 private:
  size_t size_;
  std::shared_ptr<T> data_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
};

}  // namespace tensor
}  // namespace engine
