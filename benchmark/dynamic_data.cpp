
//#include "dynamic.hpp"

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

int train_size;

GlobalController controller;
int add_times, erase_times;

Cost hits_basic_evaluation(std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset, bool using_model = true) {
    experience_t exp_chosen{};
    auto min_max = get_min_max<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end());
    auto pdf = get_pdf<KEY_TYPE, VALUE_TYPE>(dataset.begin(), dataset.end(), min_max.first, min_max.second,
                                             BUCKET_SIZE);
    std::copy(pdf.begin(), pdf.end(), exp_chosen.distribution);
    exp_chosen.data_size = float(train_size);
    Hits::Configuration conf;
    if (using_model) {
        auto best_gen = controller.get_best_action_GA(exp_chosen);
        conf = best_gen.conf;
    } else {
        conf = Hits::Configuration::default_configuration();
    }
    exp_chosen.conf = conf;
    auto index = new Hits::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);
    index->bulk_load(dataset.begin(), dataset.begin() + train_size);
    std::shuffle(dataset.begin(), dataset.begin() + train_size, e);
    tc.synchronization();
    long long opt_count = 0;
    int insert_cursor = train_size;
    int erase_cursor = 0;
    int read_times = erase_times + add_times;
    while (insert_cursor < int(dataset.size())) {
        VALUE_TYPE value;
        for (int i = 0; i < read_times; ++i) {
            opt_count++;
            auto random_index = int(e() % (insert_cursor - erase_cursor)) + erase_cursor;
            if (!index->get(dataset[random_index].first, value)
                || value != dataset[random_index].second) {
                std::cout << "hits get error:" << i << std::endl;
            }
        }
        for (int i = 0; i < erase_times; ++i) {
            opt_count++;
            if (!index->erase(dataset[erase_cursor++].first)) {
                puts("hits erase error !");
            }
            if (erase_cursor >= insert_cursor) {
                goto END;
            }
        }
        for (int i = 0; i < add_times && insert_cursor < int(dataset.size()); ++i, ++insert_cursor) {
            opt_count++;
            if (!index->add(dataset[insert_cursor].first, dataset[insert_cursor].second)) {
                puts("hits add error !");
            }
        }
    }
    END:
    exp_chosen.cost.add = float(opt_count / (double) tc.get_timer_second());
    exp_chosen.cost.memory = index->memory_occupied() / (1024 * 1024);
    delete index;
    return exp_chosen.cost;
}



int main() {
    controller.load_in();
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    std::vector<std::pair<KEY_TYPE, VALUE_TYPE>> dataset;
    auto exp_chosen = experience_t();
    std::vector<std::pair<int, int>> add_erase_proportions = {
            {0, 5},
            {1, 4},
            {2, 3},
            {3, 2},
            {4, 1},
            {5, 0},
//            {1, 4000},//for alex
//            {1, 1000},
//            {1, 250},
//            {1, 100},
    };
    std::ofstream result(father_path + "dynamic_2_result/add_erase_result_2.txt");
    for (const auto &dataset_name: std::vector<std::string>({  "osmc.data","face.data", "logn.data","uden.data",})) {
        dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
        std::shuffle(dataset.begin(), dataset.end(), e);
        train_size = int(0.2 * int(dataset.size()));
        std::sort(dataset.begin(), dataset.begin() + train_size);
        for (auto pp: add_erase_proportions) {
            add_times = pp.first;
            erase_times = pp.second;
            std::cout << MAGENTA << "dataset:" << dataset_name << " train_rate:" << std::setw(4) << pp << RESET << std::endl;
            result << "dataset:" << dataset_name << " train_rate:"<< pp << std::endl;
            exp_chosen.cost = hits_basic_evaluation(dataset, true);
            std::cout << "cha:" << exp_chosen.cost << std::endl;
            result << "cha:" << exp_chosen.cost << std::endl;
        }
    }
}
