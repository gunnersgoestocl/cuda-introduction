#include "utils.h"
#include <iostream>
#include <fstream>
#include <chrono>

void loadBinaryData(const std::string& filename, float* data, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    file.read(reinterpret_cast<char*>(data), size * sizeof(float));
    file.close();
}

void saveOutputData(const std::string& filename, const float* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    file.close();
}

double measureExecutionTime(const std::function<void()>& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}