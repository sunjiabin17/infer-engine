#include <glog/logging.h>
#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

#include "data/tensor.h"

TEST(TENSOR, TENSOR_CONSTRUCTOR1) {
  engine::tensor::Tensor<float> tensor;
  EXPECT_EQ(tensor.size(), 0);
  EXPECT_EQ(tensor.shape().size(), 0);
  EXPECT_EQ(tensor.strides().size(), 0);
  EXPECT_EQ(tensor.data(), nullptr);
}

TEST(TENSOR, TENSOR_CONSTRUCTOR2) {
  engine::tensor::Tensor<float> tensor({1, 2, 3});
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.shape().size(), 3);
  EXPECT_EQ(tensor.strides().size(), 3);
  EXPECT_NE(tensor.data(), nullptr);

  LOG(INFO) << "shape: ";
  for (auto i : tensor.shape()) {
    LOG(INFO) << i;
  }
  LOG(INFO) << "strides: ";
  for (auto i : tensor.strides()) {
    LOG(INFO) << i;
  }
}

TEST(TENSOR, TENSOR_CONSTRUCTOR3) {
  float data[6] = {1, 2, 3, 4, 5, 6};
  engine::tensor::Tensor<float> tensor({1, 2, 3}, data);
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.shape().size(), 3);
  EXPECT_EQ(tensor.strides().size(), 3);
  EXPECT_NE(tensor.data(), nullptr);

  LOG(INFO) << "shape: ";
  for (auto i : tensor.shape()) {
    LOG(INFO) << i;
  }
  LOG(INFO) << "strides: ";
  for (auto i : tensor.strides()) {
    LOG(INFO) << i;
  }
}

TEST(TENSOR, TENSOR_CONSTRUCTOR4) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  engine::tensor::Tensor<float> tensor({1, 2, 3}, data);
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.shape().size(), 3);
  EXPECT_EQ(tensor.strides().size(), 3);
  EXPECT_NE(tensor.data(), nullptr);

  LOG(INFO) << "shape: ";
  for (auto i : tensor.shape()) {
    LOG(INFO) << i;
  }
  LOG(INFO) << "strides: ";
  for (auto i : tensor.strides()) {
    LOG(INFO) << i;
  }
}

TEST(TENSOR, TENSOR_COPY_CONSTRUCTOR1) {
  engine::tensor::Tensor<float> tensor1({1, 2, 3});
  engine::tensor::Tensor<float> tensor2(tensor1);
  EXPECT_EQ(tensor1.size(), tensor2.size());
  EXPECT_EQ(tensor1.shape().size(), tensor2.shape().size());
  EXPECT_EQ(tensor1.strides().size(), tensor2.strides().size());
  EXPECT_EQ(tensor1.data(), tensor2.data());
}

TEST(TENSOR, TENSOR_COPY_CONSTRUCTOR2) {
  engine::tensor::Tensor<float> tensor1({1, 2, 3});
  engine::tensor::Tensor<float> tensor2 = tensor1;
  EXPECT_EQ(tensor1.size(), tensor2.size());
  EXPECT_EQ(tensor1.shape().size(), tensor2.shape().size());
  EXPECT_EQ(tensor1.strides().size(), tensor2.strides().size());
  EXPECT_EQ(tensor1.data(), tensor2.data());
}

TEST(TENSOR, TENSOR_MOVE_CONSTRUCTOR1) {
  engine::tensor::Tensor<float> tensor1({1, 2, 3});
  std::string address1 = reinterpret_cast<std::ostringstream &>(
                             std::ostringstream() << tensor1.data())
                             .str();

  engine::tensor::Tensor<float> tensor2(std::move(tensor1));
  std::string address2 = reinterpret_cast<std::ostringstream &>(
                             std::ostringstream() << tensor2.data())
                             .str();

  EXPECT_EQ(address1, address2);
  EXPECT_EQ(tensor1.size(), 0);
  EXPECT_EQ(tensor1.shape().size(), 0);
  EXPECT_EQ(tensor1.strides().size(), 0);
  EXPECT_EQ(tensor1.data(), nullptr);
  EXPECT_EQ(tensor2.size(), 6);
  EXPECT_EQ(tensor2.shape().size(), 3);
  EXPECT_EQ(tensor2.strides().size(), 3);
  EXPECT_NE(tensor2.data(), nullptr);
}

TEST(TENSOR, TENSOR_MOVE_CONSTRUCTOR2) {
  engine::tensor::Tensor<float> tensor1({1, 2, 3});
  std::string address1 = reinterpret_cast<std::ostringstream &>(
                             std::ostringstream() << tensor1.data())
                             .str();

  engine::tensor::Tensor<float> tensor2 = std::move(tensor1);
  std::string address2 = reinterpret_cast<std::ostringstream &>(
                             std::ostringstream() << tensor2.data())
                             .str();

  EXPECT_EQ(address1, address2);
  EXPECT_EQ(tensor1.size(), 0);
  EXPECT_EQ(tensor1.shape().size(), 0);
  EXPECT_EQ(tensor1.strides().size(), 0);
  EXPECT_EQ(tensor1.data(), nullptr);
  EXPECT_EQ(tensor2.size(), 6);
  EXPECT_EQ(tensor2.shape().size(), 3);
  EXPECT_EQ(tensor2.strides().size(), 3);
  EXPECT_NE(tensor2.data(), nullptr);
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  LOG(INFO) << "Running tests...";
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
