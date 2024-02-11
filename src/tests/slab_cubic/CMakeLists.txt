cmake_minimum_required(VERSION 3.12.0)
project(SlabCubic LANGUAGES CXX C)

include(../../../common/cmake-common.cmake)

add_executable(main main.cpp ../../BaseSolver.cpp)
deal_ii_setup_target(main)
