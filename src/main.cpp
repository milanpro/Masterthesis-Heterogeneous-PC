#include <iostream>
#include <boost/program_options.hpp>
#include "csv_parser.hpp"
#include "correlation/corOwn.cuh"
#include "util/state.cuh"
#include "independence/skeleton.hpp"
#include "loadbalance/balancer.hpp"
namespace po = boost::program_options;
#include <iostream>
#include <vector>
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
        ("row-mult", po::value<float>()->default_value(0.2), "maximum rows balanced multiplier level 0,1,4+")
        ("row-mult2", po::value<float>()->default_value(0.2), "maximum rows balanced multiplier level 2")
        ("row-mult3", po::value<float>()->default_value(0.1), "maximum rows balanced multiplier level 3")
        ("max-level,m", po::value<int>()->default_value(4), "maximum level")
        ("corr", "input file is a correlation matrix")
        ("gpus,g", po::value<vector<int>>()->multitoken(), "GPU deviceIds that should be used")
        ("thread-count,t", po::value<int>(), "number of threads used by openMP")
        ("csv-export", po::value<string>(), "Export runtimes execution metrics to CSV")
        ("gpu-only", "execution on gpu only")
        ("cpu-only", "execution on cpu only")
        ("power9-ats", "Use power9 system allocator for ATS (Address translation service)")
        ("workstealing,w", "use workstealing CPU executor")
        ("print-sepsets,p", "prints-sepsets")
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

#ifdef NDEBUG
    bool verbose = vm.count("verbose") != 0;
#else
    std::cout << "Debug mode, verbose is on." << std::endl;
    bool verbose = true;
#endif

    if (vm.count("thread-count"))
    {
        int numberOfThreads = vm["thread-count"].as<int>();
        omp_set_num_threads(numberOfThreads);
    }

    string inputFile = vm["input-file"].as<string>();
    double alpha = vm["alpha"].as<double>();
    int maxLevel = vm["max-level"].as<int>();
    bool workstealing = vm.count("workstealing");

    string csvExportFile;
    if (vm.count("csv-export"))
    {
        csvExportFile = vm["csv-export"].as<string>();
    }

    if (verbose)
    {
        cout << "Using " << omp_get_max_threads() << " OpenMP thread(s) in pool" << endl;
        cout << "Using following GPUs:" << endl;
        for (auto deviceId : gpuList) {
            cout << "\t" << deviceId << endl;
        }
        cout << "Reading file: " << inputFile << endl;
        if (csvExportFile != "") {
            cout << "Export metrics to CSV file: " << csvExportFile << endl;
        }
    }

    SkeletonCalculator skeletonCalculator = SkeletonCalculator(maxLevel, alpha, workstealing, csvExportFile, verbose);

    if (vm.count("gpu-only")) {
        skeletonCalculator.set_heterogeneity(Heterogeneity::GPUOnly);
    } else if (vm.count("cpu-only")) {
        skeletonCalculator.set_heterogeneity(Heterogeneity::CPUOnly);
    }

    if (vm.count("gpus")) {
        skeletonCalculator.set_gpu_list(vm["gpus"].as<vector<int>>());
    }


    bool print_sepsets = vm.count("print-sepsets");
    bool use_p9_ats = vm.count("power9-ats");

    if (vm.count("corr"))
    {
        if (!vm.count("observations"))
        {
            cout << "Observation count needed with correlation matrix input" << endl;
            return -1;
        }
        int observation_count = vm["observations"].as<int>();
        skeletonCalculator.add_correlation_matrix(inputFile, observation_count, use_p9_ats);
    }
    else
    {
        skeletonCalculator.add_observations(inputFile, use_p9_ats);
    }

    std::tuple<float, float, float> balancer_thresholds = {vm["row-mult"].as<float>(), vm["row-mult2"].as<float>(), vm["row-mult3"].as<float>()};

    skeletonCalculator.initialize_balancer(balancer_thresholds);

    skeletonCalculator.run();

    return 0;
}
