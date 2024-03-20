//
// Created by redamancyguy on 23-7-23.
//
#include <iostream>
#include <random>
#include<functional>
#include <iomanip>
#include <torch/torch.h>
#include "../include/DEFINE.h"
#include "../index/include/Index.hpp"
#include "../include/DataSet.hpp"
#include "../index/include/experience.hpp"
#include "../index/include/Controller.hpp"


double train_proportion = 0.5;
auto train_size = 0;
GlobalController controller;

auto hits_basic_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset,
                           bool using_model = true) {
    experience_t exp_chosen;
    auto min_max = get_min_max<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end());
    auto pdf = get_pdf<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end(),min_max.first,min_max.second,BUCKET_SIZE);
    std::copy(pdf.begin(),pdf.end(),exp_chosen.distribution);
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
    TimerClock tc;
    for (int i = 0; i < train_size; ++i) {
        if (!index->add(dataset[i].first, dataset[i].second)) {
            puts("hits add error !");
        }
    }
    std::vector<long long> cost_vector;
    cost_vector.reserve(dataset.size());
    tc.synchronization();
    int add_count = 0;
    for (int i = train_size; i < int(dataset.size()); ++i) {
        if (add_count++ % 333 == 0) {
            cost_vector.push_back(tc.get_timer_nanoSec());
            tc.synchronization();
        }
        if (!index->add(dataset[i].first, dataset[i].second)) {
            puts("hits add error !");
        }
    }
    delete index;
    return cost_vector;
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

int main() {
    TimerClock tc;
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset;
    auto exp_chosen = experience_t();
    std::ifstream tsk_file(father_path + "tsk.txt");
    std::ofstream result(father_path + "add_get_erase_result.txt", std::ios::out | std::ios::binary);
    result << "train_proportion:" << train_proportion << "============================" << std::endl;
    for (int _ = 0; !tsk_file.eof(); ++_) {
        EvaluationTask tsk;
        tsk_file >> tsk.dataset_name >> tsk.start >> tsk.length;
        tsk.length = 40000000;
        if (tsk.dataset_name.empty()) {
            break;
        }
        result << tsk.dataset_name << "  start:" << tsk.start << "  length:" << tsk.length << std::endl;
        dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + tsk.dataset_name);
        dataset.erase(dataset.begin(), dataset.begin() + tsk.start);
        dataset.erase(dataset.begin() + tsk.length, dataset.end());
        std::cout << MAGENTA << "  test_count:" << std::setw(4)
                  << double(dataset.size()) / double(1000000) << "*10**6" << RESET << std::endl;
        dataset.resize(dataset.size()/30);
        train_size = int(train_proportion * int(dataset.size()));
        std::sort(dataset.begin(), dataset.begin() + train_size);
        std::shuffle(dataset.begin() + train_size,dataset.end(),e);
//        auto cost_vec = alex_basic_evaluation(dataset);
//        std::cout <<"cost_vec.size:"<<cost_vec.size() << std::endl;
        std::ofstream data("data.data");
        std::map<std::string, std::string> color;
//        color["color"] = "blue";
//        for (auto i: cost_vec) {
//            data << i << " ";
//        }
//        data << std::endl;
        auto cost_vec = hits_basic_evaluation(dataset, false);
        std::cout <<"cost_vec.size:"<<cost_vec.size() << std::endl;
        for (auto i: cost_vec) {
            data << i << " ";
        }
        data << std::endl;
        color["color"] = "red";
        std::cout << "hits:" << exp_chosen.cost << std::endl;
        result << "hits:" << exp_chosen.cost << std::endl;
        return 0;
    }
}
