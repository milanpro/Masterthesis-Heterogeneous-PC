#include "csv_parser.hpp"
#include <vector>
#include <iostream>
#include <memory>
#include "armadillo"

std::vector<std::string> parse_header(std::ifstream& file_input, std::vector<std::string>& column_names) {
    std::string line;
    std::getline(file_input, line);

    std::stringstream lineStream(line);
    std::string cell;

    while (std::getline(lineStream, cell, ',')) {
        column_names.push_back(cell);
    }
    return column_names;
}

std::vector<std::vector<double>> read_csv(std::string filename, std::vector<std::string>& column_names) {
    std::ifstream file_input(filename);
    if (!file_input.is_open()) {
        std::cout << "Could not find file '" << filename << '\'' << std::endl;
        exit(1);
    }
    std::vector<std::vector<double>> data(1, std::vector<double>(1));
    int variables = 0;
    int observations = 0;
    double next_val;
    char c;

    if (!(file_input >> next_val)) {
        file_input.seekg(0, std::ios::beg);
        file_input.clear();

        parse_header(file_input, column_names);
        file_input >> next_val;
    }

    data[variables][observations] = next_val;

    file_input >> std::noskipws >> c;
    while (file_input.peek() != EOF) {
        if (c == ',') {
            variables++;
            if (observations == 0) {
                data.push_back(std::vector<double>());
            }
        } else if (c == '\r' || c == '\n') {
            observations++;
            variables = 0;
        }

        file_input >> next_val;
        data[variables].push_back(next_val);
        file_input >> std::noskipws >> c;
    }

    data[variables].pop_back();

    return data;
}

std::shared_ptr<arma::mat> CSVParser::read_csv_to_mat(std::string filename, std::vector<std::string>& column_names) {
    auto data = read_csv(filename, column_names);

    auto nr_observations = data[1].size();
    auto nr_variables = data.size();

    auto mat = std::make_shared<arma::mat>(nr_observations, nr_variables);

    for (int v = 0; v < (nr_variables); ++v) {
        std::memcpy(mat->colptr(v), data[v].data(), nr_observations * sizeof(double));
    }

    return mat;
}