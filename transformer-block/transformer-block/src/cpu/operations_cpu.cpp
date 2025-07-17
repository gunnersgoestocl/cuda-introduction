#include "operations_cpu.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <omp.h>

Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix C(A.rows, B.cols);
    
    #pragma omp parallel for
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < B.cols; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    
    return C;
}

Matrix softmax(const Matrix& scores) {
    Matrix result(scores.rows, scores.cols);
    
    #pragma omp parallel for
    for (size_t i = 0; i < scores.rows; ++i) {
        // 各行に対してsoftmaxを適用
        float max_val = *std::max_element(scores.data.begin() + i * scores.cols, 
                                         scores.data.begin() + (i + 1) * scores.cols);
        
        float sum = 0.0f;
        for (size_t j = 0; j < scores.cols; ++j) {
            float exp_val = std::exp(scores(i, j) - max_val);
            result(i, j) = exp_val;
            sum += exp_val;
        }
        
        // 正規化
        for (size_t j = 0; j < scores.cols; ++j) {
            result(i, j) /= sum;
        }
    }
    
    return result;
}

Matrix layer_norm(const Matrix& input, const Matrix& gamma, const Matrix& beta) {
    Matrix result(input.rows, input.cols);
    const float eps = 1e-6f;
    
    #pragma omp parallel for
    for (size_t i = 0; i < input.rows; ++i) {
        // 平均を計算
        float mean = 0.0f;
        for (size_t j = 0; j < input.cols; ++j) {
            mean += input(i, j);
        }
        mean /= input.cols;
        
        // 分散を計算
        float variance = 0.0f;
        for (size_t j = 0; j < input.cols; ++j) {
            float diff = input(i, j) - mean;
            variance += diff * diff;
        }
        variance /= input.cols;
        
        // 正規化
        float std_dev = std::sqrt(variance + eps);
        for (size_t j = 0; j < input.cols; ++j) {
            result(i, j) = gamma(0, j) * (input(i, j) - mean) / std_dev + beta(0, j);
        }
    }
    
    return result;
}

Matrix feed_forward(const Matrix& input, const Matrix& weights1, const Matrix& weights2) {
    // 第1層: input * weights1 + ReLU
    Matrix hidden = matmul(input, weights1);
    
    // ReLU活性化
    #pragma omp parallel for
    for (size_t i = 0; i < hidden.data.size(); ++i) {
        hidden.data[i] = std::max(0.0f, hidden.data[i]);
    }
    
    // 第2層: hidden * weights2
    return matmul(hidden, weights2);
}

void relu(std::vector<float>& data) {
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

Matrix transpose(const Matrix& input) {
    Matrix result(input.cols, input.rows);
    
    #pragma omp parallel for
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            result(j, i) = input(i, j);
        }
    }
    
    return result;
}