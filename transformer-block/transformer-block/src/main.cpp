#include <iostream>
#include <chrono>
#include "common/transformer_common.h"
#include "cpu/transformer_cpu.h"
#include "cuda/transformer_cuda.cuh"

void run_cpu(const std::string& input_file, const std::string& output_file) {
    // CPU実行のための関数
    TransformerCPU transformer_cpu;
    transformer_cpu.load_data(input_file);
    transformer_cpu.execute();
    transformer_cpu.save_output(output_file);
}

void run_cuda(const std::string& input_file, const std::string& output_file) {
    // CUDA実行のための関数
    TransformerCUDA transformer_cuda;
    transformer_cuda.load_data(input_file);
    transformer_cuda.execute();
    transformer_cuda.save_output(output_file);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    // CPU実行
    auto start_cpu = std::chrono::high_resolution_clock::now();
    run_cpu(input_file, output_file);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU execution time: " << cpu_duration.count() << " seconds" << std::endl;

    // CUDA実行
    auto start_cuda = std::chrono::high_resolution_clock::now();
    run_cuda(input_file, output_file);
    auto end_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cuda_duration = end_cuda - start_cuda;
    std::cout << "CUDA execution time: " << cuda_duration.count() << " seconds" << std::endl;

    return 0;
}