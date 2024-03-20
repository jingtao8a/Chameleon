//
// Created by redamancyguy on 23-7-27.
//

#ifndef HITS_DYNAMIC_HPP
#define HITS_DYNAMIC_HPP
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
double train_proportion = 0.1;
auto train_size = 0;
double write_times = 1;
double read_times = 10;

GlobalController controller;

Cost hits_basic_evaluation(const std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> &dataset,
                           bool using_model = true) {
    experience_t exp_chosen{};
    auto min_max = get_min_max<KEY_TYPE,VALUE_TYPE>(dataset.begin(),dataset.end());
    auto pdf = get_pdf<KEY_TYPE,VALUE_TYPE>(dataset.begin(), dataset.end(), min_max.first, min_max.second,BUCKET_SIZE);
    std::copy(pdf.begin(),pdf.end(),exp_chosen.distribution);
    exp_chosen.data_size = float(train_size);
    Hits::Configuration conf;
    if (using_model) {
        auto best_gen = controller.get_best_action_GA(exp_chosen);
        conf = best_gen.conf;
    } else {
        conf = Hits::Configuration::default_configuration();
        conf.root_fan_out = float(train_size);
    }
    exp_chosen.conf = conf;
    auto index = new Hits::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);
    TimerClock tc;
    index->bulk_load(dataset.begin(),dataset.begin() + train_size);
    tc.synchronization();
    long long opt_count = 0;
    int insert_cursor = train_size;
    while (insert_cursor < int(dataset.size())) {
        VALUE_TYPE value;
        for (int i = 0; i < read_times; ++i) {
            opt_count++;
            auto random_index = int(e() % train_size) + (insert_cursor - train_size);
            if (!index->get(dataset[random_index].first, value)
                || value != dataset[random_index].second) {
                std::cout << "hits get error:" << i << std::endl;
            }
        }
        for (int i = 0; i < write_times; ++i) {
            opt_count++;
            if (!index->erase(dataset[insert_cursor - train_size + i].first)) {
                puts("hits erase error !");
            }
        }
        for (int i = 0; i < write_times && insert_cursor < int(dataset.size()); ++i, ++insert_cursor) {
            opt_count++;
            if (!index->add(dataset[insert_cursor].first, dataset[insert_cursor].second)) {
                puts("hits add error !");
            }
        }
    }
    exp_chosen.cost.add = float(opt_count / (double) tc.get_timer_second());
    exp_chosen.cost.memory = index->memory_occupied() / (1024 * 1024);
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
    auto ts = std::size_t(double(dataset.size()) * proportion.first / (proportion.first + proportion.second));
    std::vector<std::size_t> indices(dataset.size());
    for (std::size_t i = 0; i < dataset.size(); i++) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), e);
    std::vector<T> train_dataset;
    train_dataset.reserve(ts);
    std::vector<T> test_dataset;
    test_dataset.reserve(dataset.size() - ts);
    for (std::size_t i = 0; i < ts; i++) {
        train_dataset.push_back(dataset[indices[i]]);
    }
    for (std::size_t i = ts; i < dataset.size(); i++) {
        test_dataset.push_back(dataset[indices[i]]);
    }
    return {train_dataset, test_dataset};
}

#endif //HITS_DYNAMIC_HPP
