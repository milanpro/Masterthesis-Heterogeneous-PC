#pragma once
#include "../loadbalance/balancer.hpp"
#include <string>
#include <tuple>
#include <vector>
#include <armadillo>

typedef std::tuple<int64_t, int64_t, std::tuple<TestResult, TestResult>> LevelMetrics;

class SkeletonCalculator
{
    int maxLevel;
    double alpha;
    bool verbose;
    bool workstealing;
    std::string csvExportFile;
    Heterogeneity heterogeneity;
    std::vector<int> gpuList;
    MMState state;
    Balancer balancer;

private:
public:
    SkeletonCalculator(int maxLevel, double alpha, bool workstealing, std::string csvExportFile, bool verbose = false);

    /**
     * Set usage of a specific or all processing units 
     */
    void set_heterogeneity(Heterogeneity heterogeneity);

    /**
     * Set gpus used for gpu side computations (first one is the main gpu)
     */
    void set_gpu_list(std::vector<int> gpuList);

    /**
     * Basic csv file parsing
     */
    std::shared_ptr<arma::mat> read_csv_file(std::string input_file);

    /**
     * Create state by using an observation matrix csv file
     */
    void add_observations(std::string input_file, bool use_p9_ats = false);

    /**
     * Create state by using a correlation matrix csv file
     */
    void add_correlation_matrix(std::string input_file, int observation_count, bool use_p9_ats = false);

    /**
     * Initialize the balancer by adding thresholds
     */
    void initialize_balancer(std::tuple<float, float, float> balancer_thresholds);

    /**
     * Run skeleton calculation
     */
    void run(bool print_sepsets);
    LevelMetrics calcLevel(int level);
    void export_metrics(std::vector<LevelMetrics> levelMetrics, int64_t executionDuration, int nrEdges);
};
