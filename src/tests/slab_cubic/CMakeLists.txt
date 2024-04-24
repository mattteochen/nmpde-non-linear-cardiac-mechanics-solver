cmake_minimum_required(VERSION 3.12.0)
project(SlabCubic LANGUAGES CXX C)

include(../../../common/cmake-common.cmake)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/
  ${CMAKE_CURRENT_SOURCE_DIR}/
)

add_executable(main main.cpp ../../lib/cardiac_mechanics/BaseSolverGuccione.cpp)
deal_ii_setup_target(main)
