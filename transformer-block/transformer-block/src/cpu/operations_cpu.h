#ifndef OPERATIONS_CPU_H
#define OPERATIONS_CPU_H

#include <vector>
#include <cstddef>

class Matrix {
public:
    std::vector<float> data;
    size_t rows, cols;
    
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0f) {}
    Matrix() : rows(0), cols(0) {}
    
    float& operator()(size_t i, size_t j) { return data[i * cols + j]; }
    const float& operator()(size_t i, size_t j) const { return data[i * cols + j]; }
    
    size_t size() const { return data.size(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

// Function declarations for CPU operations
Matrix matmul(const Matrix& A, const Matrix& B);
Matrix softmax(const Matrix& scores);
Matrix layer_norm(const Matrix& input, const Matrix& gamma, const Matrix& beta);
Matrix feed_forward(const Matrix& input, const Matrix& weights1, const Matrix& weights2);

// Additional utility functions for transformer
void relu(std::vector<float>& data);
Matrix transpose(const Matrix& input);

#endif // OPERATIONS_CPU_H