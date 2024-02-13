/**
 * @file Reporter.hpp
 * @brief Header file containing the reporter class for managing
 * test reporting into log file.
 */

#ifndef REPORTER_HPP
#define REPORTER_HPP

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

/**
 * @class Reporter
 * @brief Class for managing test result reports to output file.
 */
class Reporter {
private:
  std::string m_report_file_name; /**< The name of the report file. */
  std::string m_header;           /**< The log header (CSV columns) */
  std::string m_problem_name;     /**< The name of the problem. */
  std::ofstream m_report_file;    /**< Output file stream for reporting. */

public:
  /**
   * @brief Default constructor.
   */
  Reporter() = default;
  /**
   * @brief Constructor.
   * @param report_file_name The name of the report file.
   * @param header The report file columns tags.
   * @param problem_name The name of the problem.
   */
  Reporter(const std::string report_file_name, const std::string header,
           const std::string problem_name)
      : m_report_file_name(report_file_name), m_header(header),
        m_problem_name(problem_name) {}
  /**
   * @brief Write data to the report file.
   * @tparam Var Variable argument types.
   * @param vars Variable arguments to write to the report file.
   */
  template <typename... Var> void write(Var... vars) {
    std::streamsize report_file_size;
    std::ifstream report_file(m_report_file_name, std::ios::ate);
    if (report_file.is_open()) {
      report_file_size = report_file.tellg();
      report_file.close();
    } else {
      report_file_size = 0;
    }
    m_report_file.open(m_report_file_name, std::ios::app);
    if (!m_report_file.is_open()) {
      std::cerr << "Failed to open output file" << std::endl;
      return;
    }

    // write header
    if (report_file_size == 0) {
      m_report_file << m_header << std::endl;
    }

    // write content
    m_report_file << m_problem_name << ',';
    ((m_report_file << vars), ...) << std::endl;
    m_report_file.close();
  }
};

#endif // REPORTER_HPP
