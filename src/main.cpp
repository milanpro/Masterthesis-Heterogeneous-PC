#include <iostream>
#include <boost/program_options.hpp>
#include "csv_parser.hpp"
#include "correlation/corOwn.cuh"
#include "util/State.cuh"
#include "independence/skeleton.cuh"
namespace po = boost::program_options;
#include <iostream>
#include <iterator>
using namespace std;

bool VERBOSE;

int main(int argc, char const *argv[])
{

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce a help message")
        ("input-file,i", po::value<string>()->default_value("../../data/cooling_house.csv"), "input file")
        ("alpha,a", po::value<double>()->default_value(0.05), "alpha value")
        ("observations,o", po::value<int>(), "observation count")
        ("max-level,m", po::value<int>()->default_value(4), "maximum level")
        ("corr", "input file is a correlation matrix")
        ("verbose,v", "verbose output");

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    if (vm.count("help"))
    {
        cout << desc;
        return 0;
    }
    try
    {
        po::notify(vm);
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    string inputFile = vm["input-file"].as<string>();
    double alpha = vm["alpha"].as<double>();
    int maxLevel = vm["max-level"].as<int>();

     #ifdef NDEBUG
        bool verbose = vm.count("verbose") != 0;
     #else
        bool verbose = true;
    #endif

    VERBOSE = verbose;
    if (verbose) {
        cout << "Reading file: " << inputFile << endl;
    }

    string _match(inputFile);
    shared_ptr<arma::mat> array_data;
    vector<string> column_names(0);

    if (_match.find(".csv") != string::npos)
    {
        array_data = CSVParser::read_csv_to_mat(inputFile, column_names);
    }
    else
    {
        cout << "Cannot process file '" << inputFile << "\'." << endl;
        cout << "Has to be .csv format." << endl;
        return -1;
    }

    if (vm.count("corr"))
    {
        if (!vm.count("observations"))
        {
            cout << "Observation count needed with correlation matrix input" << endl;
            return -1;
        }

        MMGPUState state = MMGPUState(array_data.get()->n_cols, vm["observations"].as<int>(), alpha, maxLevel);
        memcpy(state.cor, array_data.get()->begin(), state.p * state.p * sizeof(double));
        calcSkeleton(&state, 1);
    }
    else
    {
        MMGPUState state = MMGPUState(array_data.get()->n_cols, array_data.get()->n_rows, alpha, maxLevel);
        gpuPMCC(array_data.get()->begin(), state.p, state.observations, state.cor);
        calcSkeleton(&state, 1);
    }

    return 0;
}
