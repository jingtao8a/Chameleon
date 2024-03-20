//
// Created by redamancyguy on 23-4-18.
//

#ifndef HITS_STANDARDSCALAR_HPP
#define HITS_STANDARDSCALAR_HPP

#include <cstdlib>
#include <vector>
#include <string>
#include <queue>
#include "DEFINE.h"

//class StandardScalar{
//public:
//    float first_moment = 0;
//    float var = 1;
//
//    void save(const std::string &filename){
//        auto *file=::fopen((model_father_path+filename).c_str(), "w");
//        if(std::fwrite(&first_moment, sizeof(double), 1, file)==0){
//            throw MyException("load Standard fail !");
//        }
//        if(std::fwrite(&var, sizeof(double), 1, file)==0){
//            throw MyException("load Standard fail !");
//        }
//        std::fclose(file);
//    }
//
//    void load(const std::string &filename){
//        auto *file=::fopen((model_father_path+filename).c_str(), "r");
//        if(std::fread(&first_moment, sizeof(double), 1, file)==0){
//            throw MyException("save Standard fail !");
//        }
//        if(std::fread(&var, sizeof(double), 1, file)==0){
//            throw MyException("save Standard fail !");
//        }
//        std::fclose(file);
//    }
//
//
//    void fit(const std::vector<float>& data){
//        std::priority_queue<float, std::vector<float>, std::greater<>> data_queue;
//        for(auto i: data){
//            data_queue.push(i);
//        }
//        while(data_queue.size()>1){
//            float a=data_queue.top();
//            data_queue.pop();
//            float b=data_queue.top();
//            data_queue.pop();
//            data_queue.push(a+b);
//        }
//        first_moment=data_queue.top()/float(data.size());
//        data_queue.pop();
//        assert(data_queue.empty());
//        for(auto i: data){
//            data_queue.push((i-first_moment)*(i-first_moment));
//        }
//        while(data_queue.size()>1){
//            float a=data_queue.top();
//            data_queue.pop();
//            float b=data_queue.top();
//            data_queue.pop();
//            data_queue.push(a+b);
//        }
//        var=data_queue.top()/float(data.size());
//        var=float(std::pow(var, 0.5));
//        if(var < 1e-20){
//            var = 1e-20;
//        }
//    }
//
//    [[nodiscard]] float forward(float data) const{
//        return (data-first_moment)/var;
//    }
//
//    [[nodiscard]] float inverse(float data) const{
//        return data*var+first_moment;
//    }
//
//    [[nodiscard]] std::vector<float> forward(std::vector<float> data) const{
//        for(auto &i: data){
//            i=(i-first_moment)/var;
//        }
//        return data;
//    }
//
//    [[nodiscard]] std::vector<float> inverse(std::vector<float> data) const{
//        for(auto &i: data){
//            i=i*var+first_moment;
//        }
//        return data;
//    }
//
//    [[nodiscard]] torch::Tensor forward(const torch::Tensor &data) const {
//        return data.sub(first_moment).div(var);
//    }
//
//
//    [[nodiscard]] torch::Tensor inverse(const torch::Tensor &data) const {
//        return data.mul(var).add(first_moment);
//    }
//};

#endif //HITS_STANDARDSCALAR_HPP
