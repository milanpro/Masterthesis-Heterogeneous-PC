#include <iostream>
#include <boost/program_options.hpp>
#include "csv_parser.hpp"
#include "correlation/corOwn.cuh"
#include "util/state.cuh"
#include "independence/skeleton.hpp"
namespace po = boost::program_options;
#include <iostream>
#include <iterator>
#include <omp.h>
using namespace std;

#ifdef __linux__ 
  const string DEFAULT_INPUT_FILE = "../../data/cooling_house.csv";
#elif _WIN32
    const string DEFAULT_INPUT_FILE = "../../../data/cooling_house.csv";
#endif

int main(int argc, char const *argv[])
{

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce a help message")
        ("input-file,i", po::value<string>()->default_value(DEFAULT_INPUT_FILE), "input file")
        ("alpha,a", po::value<double>()->default_value(0.05), "alpha value")
        ("observations,o", po::value<int>(), "observation count")
        ("max-level,m", po::value<int>()->default_value(4), "maximum level")
        ("corr", "input file is a correlation matrix")
        ("gpu-count,g", po::value<int>()->default_value(1), "number of gpus used")
        ("thread-count,t", "number of threads used by openMP")
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
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    string inputFile = vm["input-file"].as<string>();
    double alpha = vm["alpha"].as<double>();
    int maxLevel = vm["max-level"].as<int>();
    int numberOfGPUs = vm["gpu-count"].as<int>();

    if (vm.count("thread-count")) {
        int numberOfThreads = vm["thread-count"].as<int>();
        omp_set_num_threads(numberOfThreads);
    }

#ifdef NDEBUG
    bool verbose = vm.count("verbose") != 0;
#else
    bool verbose = true;
#endif

    if (verbose)
    {
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

    if (omp_get_max_threads() > array_data.get()->n_cols) {
        omp_set_num_threads((int) array_data.get()->n_cols);
    }

    if (vm.count("corr"))
    {
        if (!vm.count("observations"))
        {
            cout << "Observation count needed with correlation matrix input" << endl;
            return -1;
        }

        MMState state = MMState(array_data.get()->n_cols, vm["observations"].as<int>(), alpha, maxLevel);
        memcpy(state.cor, array_data.get()->begin(), state.p * state.p * sizeof(double));
        calcSkeleton(&state, numberOfGPUs, verbose);
    }
    else
    {
        MMState state = MMState(array_data.get()->n_cols, (int) array_data.get()->n_rows, alpha, maxLevel);
        gpuPMCC(array_data.get()->begin(), state.p, state.observations, state.cor, verbose);
        calcSkeleton(&state, numberOfGPUs, verbose);
    }

    return 0;
}
