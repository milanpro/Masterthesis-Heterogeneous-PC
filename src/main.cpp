#include <iostream>
#include "csv_parser.hpp"
#include "./State.cuh"
int main(int argc, char const *argv[])
{
    const char* filename;
    double alpha = 0.01;

    if (argc == 2 || argc == 3) {
        filename = argv[1];
        if (argc == 3) {
            std::istringstream s2(argv[3]);
            if (!(s2 >> alpha)) std::cerr << "Invalid number " << argv[3] << '\n';
        }
    } else {
        filename = "../../data/cooling_house.csv";
/*         std::cout << "Usage: ./heterogpc <filename> [alpha=0.01]"
                  << std::endl;
        return 1; */
    }

    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.precision(10);

    std::string _match(filename);
    std::shared_ptr<arma::mat> array_data;
    std::vector<std::string> column_names(0);

    if (_match.find(".csv") != std::string::npos) {
        array_data = CSVParser::read_csv_to_mat(filename, column_names);
    } else {
        std::cout << "Cannot process file '" << filename << "\'." << std::endl;
        std::cout << "Has to be .csv format." << std::endl;

        return 1;
    }

    uint64_t p = array_data.get()->n_cols;
    uint64_t observations = array_data.get()->n_rows;
    
    State state = State(p, observations, alpha, 4);
    
    std::cout << state.p << "\n";
    return 0;
}
