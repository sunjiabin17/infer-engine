#include <gtest/gtest.h>
#include <glog/logging.h>


TEST(Test, Test1) {
  EXPECT_EQ(1, 1);
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  LOG(INFO) << "Running tests...";
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
