# test

find_package(GTest REQUIRED)
find_package(glog REQUIRED)

add_executable(test test.cc)
target_link_libraries(test GTest::GTest glog::glog)


add_executable(test_tensor test_tensor.cc)
target_include_directories(test_tensor PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_tensor GTest::GTest glog::glog tensor)
