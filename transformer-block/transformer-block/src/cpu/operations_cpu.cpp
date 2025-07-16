#include "operations_cpu.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

void matmul(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int N, int K) {
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0;
            for (int k = 0; k < K; ++k) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void softmax(const std::vector<float>& scores, std::vector<float>& output, int size) {
    float max_score = *std::max_element(scores.begin(), scores.end());
    float sum_exp = 0.0f;

    for (int i = 0; i < size; ++i) {
        output[i] = std::exp(scores[i] - max_score);
        sum_exp += output[i];
    }

    for (int i = 0; i < size; ++i) {
        output[i] /= sum_exp;
    }
}

void layer_norm(const std::vector<float>& input, std::vector<float>& output, float epsilon) {
    float mean = 0.0f;
    float variance = 0.0f;
    int size = input.size();

    for (int i = 0; i < size; ++i) {
        mean += input[i];
    }
    mean /= size;

    for (int i = 0; i < size; ++i) {
        variance += (input[i] - mean) * (input[i] - mean);
    }
    variance /= size;

    float stddev = std::sqrt(variance + epsilon);

    for (int i = 0; i < size; ++i) {
        output[i] = (input[i] - mean) / stddev;
    }
}

void feed_forward(const std::vector<float>& input, const std::vector<float>& weights1, const std::vector<float>& weights2, std::vector<float>& output, int input_size, int hidden_size) {
    std::vector<float> hidden(hidden_size);
    matmul(input, weights1, hidden, 1, hidden_size, input_size);
    
    #pragma omp parallel for
    for (int i = 0; i < hidden_size; ++i) {
        hidden[i] = std::max(0.0f, hidden[i]); // ReLU activation
    }

    matmul(hidden, weights2, output, 1, input_size, hidden_size);
}

void transformer_block(const std::vector<float>& input_embeddings, const std::vector<float>& weights_q, const std::vector<float>& weights_k, const std::vector<float>& weights_v, const std::vector<float>& ff_weights1, const std::vector<float>& ff_weights2, std::vector<float>& output) {
    int input_size = 256; // Input dimension
    int hidden_size = 768; // Intermediate dimension
    std::vector<float> Q(input_size), K(input_size), V(input_size);
    std::vector<float> scores(hidden_size), attention_scores(hidden_size), context(hidden_size), ff_output(input_size);

    // Compute Q, K, V
    matmul(input_embeddings, weights_q, Q, 1, hidden_size, input_size);
    matmul(input_embeddings, weights_k, K, 1, hidden_size, input_size);
    matmul(input_embeddings, weights_v, V, 1, hidden_size, input_size);

    // Compute attention scores
    matmul(Q, K, scores, 1, hidden_size, hidden_size);
    softmax(scores, attention_scores, hidden_size);
    
    // Compute context
    matmul(attention_scores, V, context, 1, hidden_size, hidden_size);

    // Residual connection
    for (int i = 0; i < input_size; ++i) {
        output[i] = input_embeddings[i] + context[i];
    }

    // Layer normalization
    std::vector<float> norm_output(input_size);
    layer_norm(output, norm_output, 1e-6);

    // Feed forward
    feed_forward(norm_output, ff_weights1, ff_weights2, output, input_size, hidden_size);
}