#include <vector>
#include <iostream>
#include <memory>
#include "armadillo"

class CSVParser {
    public:
        static std::shared_ptr<arma::mat> read_csv_to_mat(const char* filename, std::vector<std::string>& column_names);
};
