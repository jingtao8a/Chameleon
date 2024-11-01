//
// Created by redamancyguy on 23-5-18.
//
#include <iostream>
#include <unistd.h>

class RunningStatus {
public:
    double random_rate = 1;
    long long exp_num = 0;//experience目录下sample生成的下一个exp文件的编号
    char buffer[4096]{};
};
#define using_small_network
#include "../../include/DEFINE.h"
#include "../include/Parameter.h"
#include "../include/Controller.hpp"
#include "../include/Index.hpp"
#include <c10/cuda/CUDAGuard.h>

#define sample_batch 5
#define using_model

static TimerClock tc;

[[noreturn]] void sample(int pid){
    auto *rs = get_shared_memory<RunningStatus>();//用于进程间通信
    std::vector<experience_t> exps_local;
#ifdef using_model
    GlobalController controller;
#endif
    auto exp_chosen = experience_t();
    TimerClock tc_speed;
    std::vector<std::string> dataset_names = scanFiles(data_father_path);
    auto file = std::fopen((experience_father_path + std::to_string(rs->exp_num++) + ".exp").c_str(), "w");
    while(true){
#ifdef using_model
        controller.load_in();
#endif
        for (int _ = 0; _ < sample_batch; ++_) {
#ifdef using_model
            controller.random_weight();
#endif
            Hits::inner_cost = 0;
            Hits::leaf_cost = 0;
            int random_length = shrink_dataset_size(int(std::pow(random_u_0_1(), 1) * double(max_data_set_size - min_data_set_size)) + min_data_set_size);
            //随机选取一个数据集存储到dataset
            auto dataset_name = dataset_names[e() % dataset_names.size()];
            auto dataset = dataset_source::get_dataset<std::pair<KEY_TYPE, VALUE_TYPE>>(data_father_path + dataset_name);
            std::shuffle(dataset.begin(), dataset.end(),e);

            //随机截断dataset并排序，阶段后的dataset的长度为random_length
            random_length = std::min(random_length, int(float(dataset.size()) * float(0.9 + random_u_0_1() * 0.1)));
            auto random_start = (int) (e() % (dataset.size() - random_length));
            std::sort(dataset.begin() + random_start, dataset.begin() + random_start + random_length,
                      [=](std::pair<KEY_TYPE, VALUE_TYPE> &a,std::pair<KEY_TYPE, VALUE_TYPE> &b){return a.first < b.first;});

            //将dataset的pdf 存储到exp_chosen的distribution
            //将dataset的size 存储到exp_chosen的data_size中
            auto min_max = get_min_max<KEY_TYPE,VALUE_TYPE>(dataset.begin() + random_start, dataset.begin() + random_start + random_length);
            auto pdf = get_pdf<KEY_TYPE,VALUE_TYPE>(dataset.begin() + random_start, dataset.begin() + random_start + random_length, min_max.first, min_max.second,BUCKET_SIZE);
            std::copy(pdf.begin(),pdf.end(),exp_chosen.distribution);
            exp_chosen.data_size = float(random_length);

            //随机生成一个conf 存储到exp_chosen的conf中
            auto conf = Hits::Configuration::random_configuration();
            exp_chosen.conf = conf;

            //构建Hits
            auto index = new Hits::Index<KEY_TYPE, VALUE_TYPE>(conf, min_max.first, min_max.second);
            index->bulk_load(dataset.begin() + random_start, dataset.begin() + random_start + random_length);

            //计算memory和get_cost 存储到 exp_chosen的cost中
            auto memory = index->memory_occupied() / float(float(random_length) *(sizeof(Hits::DataNode<KEY_TYPE,VALUE_TYPE>::slot_type)));
            VALUE_TYPE value;
            Hits::inner_cost = 0;
            for (auto i=dataset.begin() + random_start, end=dataset.begin() + random_start + random_length;i < end;++i) {
                if (!index->get_with_cost(i->first, value)) {//计算get操作开销
                    std::cout << "get error:" << i->first << std::endl;
                }
            }
            auto get_cost = float(float(Hits::inner_cost) * inner_cost_weight + float(Hits::leaf_cost) * leaf_cost_weight) / ((float) random_length);
            exp_chosen.cost.memory = memory;
            exp_chosen.cost.get = get_cost;
            delete index;

            std::cout <<"memory:query:"<<controller.memory_weight<<":"<<controller.query_weight;
            std::cout << BLUE <<" pid:"<<pid << " speed : " << "   "<< 3600 / tc_speed.get_timer_second() << "/hour  " << RESET;
            tc_speed.synchronization();
            std::cout  << MAGENTA << "  length:" << std::setw(4) << double(random_length) / double(1000000) << "*10**6"
                       << "  root size:" << std::setw(10) << conf.root_fan_out
                       <<"  inner size:"<<conf.fan_outs[0][0]<<" "<<conf.fan_outs[0][INNER_FANOUT_COLUMN/2]<< RESET << std::endl;
            std::cout <<RED<<"pid:"<<pid<<"   Cost:"<<exp_chosen.cost<<RESET<< std::endl;
            exps_local.push_back(exp_chosen);
        }
        for(auto i:exps_local){
            std::fwrite(&i, sizeof(experience_t), 1, file);
        }
        exps_local.clear();
    }
}

[[noreturn]] void train(){
    auto q_model = std::make_shared<Global_Q_network>(Global_Q_network());
    q_model->to(GPU_DEVICE);
    q_model->train();
    auto q_optimizer = torch::optim::Adam(q_model->parameters(),
                                          torch::optim::AdamOptions(train_lr).
                                                  weight_decay(train_wd));
    int break_times = 10;
    ExpGenerator exp_g;
    RewardScalar scalar(2);
    for(int steps = 0;;++steps){
        float loss_count = 0;
        for (int flush_count = 0;flush_count < break_times; flush_count++) {
            for (int i = 0; i < train_steps; ++i) {
                auto exp_tensor_batch = exp_g.exp_batch(BATCH_SIZE);
                auto pdf = std::get<0>(exp_tensor_batch).to(GPU_DEVICE);
                auto value = std::get<1>(exp_tensor_batch).to(GPU_DEVICE);
                auto root_fanout = std::get<2>(exp_tensor_batch).to(GPU_DEVICE);
                auto inner_fanout = std::get<3>(exp_tensor_batch).to(GPU_DEVICE);
                auto reward = std::get<4>(exp_tensor_batch).to(GPU_DEVICE);
                scalar.forward_and_fit(reward);
                auto pred = q_model->forward(pdf, value, root_fanout, inner_fanout);
                pred = scalar.inverse(pred);
                auto loss = torch::nn::L1Loss()->forward(pred,reward);
                q_optimizer.zero_grad();
                loss.backward();
                q_optimizer.step();
                fflush(stdout);
                tc.synchronization();
                loss_count += loss.to(CPU_DEVICE).item().toFloat();
                printf("loss:%f\r",loss.to(CPU_DEVICE).item().toFloat());
            }
        }
        std::cout << GREEN <<"avg loss:"<<float(loss_count / float(train_steps * break_times))<<RESET<<std::endl;
        q_model->to(CPU_DEVICE);
        torch::save(q_model, q_model_path);
        scalar.save(q_scalar_path);
        q_model->to(GPU_DEVICE);
        sleep(3);
    }
}
int main(int argc, char const *argv[]) {

    double random_rate_discount_rate = 0.9997;

    auto *rs = create_shared_memory<RunningStatus>();//共享内存的方式创建 RunningStatus
    rs->random_rate = 1;
    std::cout << "rs->random_rate:" <<rs->random_rate << std::endl;
    const int process_count = 3;
    const int process_count2= 5;

    rs->exp_num = max_exp_number(scanFiles(experience_father_path)) + 1;

    int pid = 0;

    GPU_DEVICE = torch::Device(torch::DeviceType::CUDA,1);
    for ( auto i = 0; i < process_count; i++) {//总共开启3个进程
        random_seed();
        if (fork() == 0) {
            //开启一个sample进程
            random_seed();
            sample(pid);
        }
        ++pid;
        std::cout << "gen pid : " << pid << std::endl;
        usleep(100000);
        random_seed();
    }
    random_seed();
    GPU_DEVICE = torch::Device(torch::DeviceType::CUDA,0);
    for (auto i = 0; i < process_count2; i++) {//总共开启5个进程
        random_seed();
        if (fork() == 0) {
            //开启一个sample进程
            random_seed();
            sample(pid);
        }
        ++pid;
        std::cout << "gen pid : " << pid << std::endl;
        usleep(100000);
        random_seed();
    }
    //主进程等待采样到100个experience_t
    for(auto sample_size=count_exp(scanFiles(experience_father_path)) ;sample_size< 100;sample_size = count_exp(scanFiles(experience_father_path))){
        std::cout <<"waiting for more samples !  --->"<<sample_size<<std::endl;
        sleep(1);
    }
    if(fork() == 0){//开启训练进程
        puts("start training !");
        train();
    }

    int samples_num = 0;
    while(rs->random_rate > 3e-3){
        if(samples_num * sample_batch < count_exp(scanFiles(experience_father_path))){
            rs->random_rate *= random_rate_discount_rate;
            ++samples_num;
            std::cout <<"rs->random_rate:"<<rs->random_rate<< std::endl;
            continue;
        }
        std::cout << "  rr:" << rs->random_rate << "  samples:" << samples_num * sample_batch << std::endl;
        sleep(10);
    }
    return 0;
}