#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

// Function to read binary data from a file
template <typename T>
void readBinaryFile(const std::string& filename, std::vector<T>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    data.resize(size / sizeof(T));
    file.read(reinterpret_cast<char*>(data.data()), size);
    file.close();
}

// Function to measure execution time
class Timer {
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed() const {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

#endif // UTILS_H