/**
 * @file Assert.hpp
 * @brief Header file containing an assertion macro
 */

#ifndef ASSERT_HPP
#define ASSERT_HPP

#include <iostream>

#define ASSERT(condition, message)                                       \
  do {                                                                   \
    if (!(condition)) {                                                  \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__   \
                << " line " << __LINE__ << ": " << message << std::endl; \
      std::terminate();                                                  \
    }                                                                    \
  } while (false)

#endif
