
#include <iostream>
#include <random>
#include<functional>
#include <iomanip>
#include <torch/torch.h>
#include "../include/DEFINE.h"

//#define CB
#define using_small_network

#include "../index/include/Index.hpp"
#include "../include/DataSet.hpp"
#include "../index/include/experience.hpp"
#include "../index/include/Controller.hpp"


std::vector<int> query_dis;

static TimerClock tc;

std::vector<int> Zipf_GenData(int n) {
    std::vector<int> result;
    std::random_device rd;
    std::array<int, std::mt19937::state_size> seed_data{};
    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    auto engine = std::mt19937{seq};
    std::vector<double> probabilities;
    const double a = 0.1;

    for (int i = 1; i <= n; i++) {
        probabilities.push_back(1.0 / pow(i, a));
    }

    std::discrete_distribution<> di(probabilities.begin(), probabilities.end());

    for (int i = 0; i < n; i++) {
        result.push_back(di(engine));
    }
    return result;
}

std::vector<int> Uniform_GenData(int n) {
    std::vector<int> result;
    auto tmp_e = e;
    tmp_e.seed(1000);
    for (int i = 0; i < n; i++) {
        result.push_back(tmp_e() % n);
    }
    return result;
}
//#define bulkload_time
auto global_timer=TimerClock();

GlobalController controller;
Cost hits_basic_evaluation(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset, bool using_model = true) {
    experience_t exp_chosen;
    auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());
    auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end(), min_max.first,
                                             min_max.second,  BUCKET_SIZE);
    std::copy(pdf.begin(), pdf.end(), exp_chosen.distribution);
    exp_chosen.data_size = float(dataset.size());
    Hits::Configuration conf;
    if (using_model) {
        auto best_gen = controller.get_best_action_GA(exp_chosen);
        conf = best_gen.conf;
    } else {
        conf = Hits::Configuration::default_configuration();
    }
    exp_chosen.conf = conf;
    auto index = new Hits::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);
    global_timer.synchronization();
    index->bulk_load(dataset.begin(), dataset.end());
#ifdef bulkload_time
    std::cout<<global_timer.get_timer_nanoSec()<<std::endl;
    delete index;
    return exp_chosen.cost;
#endif
    exp_chosen.cost.memory = float(index->memory_occupied() / (1024. * 1024.));
    auto tmp_e = e;
    tmp_e.seed(1000);
    tc.synchronization();
    VALUE_TYPE value;
    Hits::inner_cost = 0;
    Hits::leaf_cost = 0;
    for (auto id: query_dis) {
        if (!index->get_with_cost(dataset[id].first, value) || dataset[id].second != value) {
            std::cout << "cha get error:" << dataset[id].first << std::endl;
        }
    }
   // std::cout <<"avg height:"<<double(Hits::inner_cost) / double(query_dis.size()) << std::endl;
  //  std::cout <<"avg error:"<<double(Hits::leaf_cost) / double(query_dis.size()) << std::endl;
    //std::cout <<"max height:"<<3 << std::endl;
    //std::cout <<"max error:"<<double(Hits::leaf_max_cost) << std::endl;
    exp_chosen.cost.get = (float) ((double) tc.get_timer_nanoSec() / ((double) query_dis.size()));
    auto count_node_result = index->count_node_of_each_layer();
    //std::cout <<count_node_result<< std::endl;
    delete index;
    return exp_chosen.cost;
}


class EvaluationTask {
public:
    std::string dataset_name;
    int start = 0;
    int length = 0;
};

template<typename T>
std::pair<std::vector<T>, std::vector<T>> split_dataset(std::vector<T> dataset, std::pair<double, double> proportion) {
    auto train_size = std::size_t(double(dataset.size()) * proportion.first / (proportion.first + proportion.second));
    std::vector<std::size_t> indices(dataset.size());
    for (std::size_t i = 0; i < dataset.size(); i++) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), e);
    std::vector<T> train_dataset;
    train_dataset.reserve(train_size);
    std::vector<T> test_dataset;
    test_dataset.reserve(dataset.size() - train_size);
    for (std::size_t i = 0; i < train_size; i++) {
        train_dataset.push_back(dataset[indices[i]]);
    }
    for (std::size_t i = train_size; i < dataset.size(); i++) {
        test_dataset.push_back(dataset[indices[i]]);
    }
    return {train_dataset, test_dataset};
}


//tree's height
void exp5(){

    controller.load_in();
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset;
    auto exp_chosen = experience_t();
    std::string dis_type;
    std::ofstream result("buffer.txt", std::ios::out | std::ios::binary);
    for (int length: std::vector<float>({20e6, 40e6})) {
//        for(int length:std::vector<int>({200000,400000})){
        for (const auto &dataset_name: std::vector<std::string>(
                {"osmc.data","uden.data", "local_skew.data","face.data",})) {//"uden.data",  ,"wiki.data",
            dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
            if(dataset.size() > length){
                dataset.resize(length);
            }
            std::cout << "length:" << length<< " dis:" << dis_type;
            std::cout << " dataset_name:" << dataset_name << std::endl;
            result << "length:" << length << " dis:" << dis_type;
            result << " dataset_name:" << dataset_name << std::endl;
            std::sort(dataset.begin(), dataset.end(),
                      [&](std::pair<KEY_TYPE, VALUE_TYPE> &a, std::pair<KEY_TYPE, VALUE_TYPE> &b) {
                          return a.first < b.first;});
            exp_chosen.cost = hits_basic_evaluation(dataset, true);
            std::cout << "cha:" << exp_chosen.cost << std::endl;
            result << "cha:" << exp_chosen.cost << std::endl;
            puts("============================");
            result << "============================" << std::endl;
        }
    }
}

//exp1
int main() {
    controller.load_in();
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset;
    auto exp_chosen = experience_t();
    std::ifstream tsk_file(father_path + "tsk.txt");
    if (IsFileExist((father_path + "read_only/read_only_result.txt").c_str())) {
        std::time_t t = std::time(nullptr);
        std::stringstream string_s;
        string_s << std::put_time(std::localtime(&t), "%Y-%m-%d %H.%M.%S");
        rename((father_path + "read_only/read_only_result.txt").c_str(),
               (father_path + "read_only/read_only_result" + string_s.str() + ".txt").c_str());
    }
    std::ofstream result(father_path + "read_only/read_only_result.txt", std::ios::out | std::ios::binary);
    std::string dis_type;
    for (int dis = 1; dis < 2; dis++) {
        for (int length: std::vector<float>({50e6,100e6,150e6,200e6})) {
            for (const auto &dataset_name: std::vector<std::string>(
                    {"logn.data","uden.data","osmc.data","face.data"})) {
                dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
                if(dataset.size() > length){
                    dataset.resize(length);
                }
                std::sort(dataset.begin(), dataset.end(),
                          [&](std::pair<KEY_TYPE, VALUE_TYPE> &a, std::pair<KEY_TYPE, VALUE_TYPE> &b) {
                              return a.first < b.first;});
                query_dis.clear();
                if (dis == 0) {
                    query_dis = Zipf_GenData(dataset.size());
                    dis_type = "zipf";
                    std::cout << "generated zipf query distribution" << std::endl;
                } else if (dis == 1) {
                    query_dis = Uniform_GenData(dataset.size());
                    dis_type = "uniform";
                    std::cout << "generated uniform query distribution" << std::endl;
                }

                std::cout << "length:" << length<< " dis:" << dis_type;
                std::cout << " dataset_name:" << dataset_name << std::endl;
                result << "length:" << length << " dis:" << dis_type;
                result << " dataset_name:" << dataset_name << std::endl;
                std::sort(dataset.begin(), dataset.end(),
                          [&](std::pair<KEY_TYPE, VALUE_TYPE> &a, std::pair<KEY_TYPE, VALUE_TYPE> &b) {
                              return a.first < b.first;});
                exp_chosen.cost = hits_basic_evaluation(dataset, true);
                std::cout << "cha:" << exp_chosen.cost << std::endl;
                result << "cha:" << exp_chosen.cost << std::endl;
                puts("============================");
                result << "============================" << std::endl;
            }
        }
    }
}

