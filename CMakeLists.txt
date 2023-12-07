cmake_minimum_required(VERSION 3.10)

project(tllm LANGUAGES CUDA CXX)

include_directories(./include)
file(GLOB SRC ./src/*)

add_definitions("-g")
# add_definitions(-DEVAL)
# add_executable(test_tensor test_tensor.cpp ${SRC})
# add_executable(test_exp test_exp.cpp ${SRC})
# target_link_libraries (test_exp pthread) 
add_executable(train train.cpp ${SRC})
target_link_libraries (train pthread) 