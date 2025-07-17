#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <stdexcept>  // std::invalid_argument用
#include <cstdlib>    // rand(), srand()用
#include <ctime> 
#include "operations_cpu.h"
#include "utils.h"

Matrix vector_to_matrix(const std::vector<float>& vec, size_t rows, size_t cols) {
    Matrix mat(rows, cols);
    if (vec.size() != rows * cols) {
        throw std::invalid_argument("Vector size doesn't match matrix dimensions");
    }
    mat.data = vec;
    return mat;
}

std::vector<float> matrix_to_vector(const Matrix& mat) {
    return mat.data;
}

void transformer_block_cpu(const Matrix& input_embeddings,
                           const Matrix& weights_q,
                           const Matrix& weights_k,
                           const Matrix& weights_v,
                           const Matrix& ff_weights1,
                           const Matrix& ff_weights2,
                           Matrix& output) {
    
    // Step 1: Compute Q, K, V
    Matrix Q = matmul(input_embeddings, weights_q);
    Matrix K = matmul(input_embeddings, weights_k);
    Matrix V = matmul(input_embeddings, weights_v);

    // Step 2: Compute attention scores (Q * K^T)
    Matrix K_transpose = transpose(K);
    Matrix scores = matmul(Q, K_transpose);
    
    // Scale by sqrt(d_k)
    float scale = 1.0f / std::sqrt(static_cast<float>(weights_q.cols));
    for (size_t i = 0; i < scores.data.size(); ++i) {
        scores.data[i] *= scale;
    }

    // Step 3: Apply softmax to scores
    Matrix attention_scores = softmax(scores);

    // Step 4: Compute context vector (attention_scores * V)
    Matrix context = matmul(attention_scores, V);

    // Step 5: Residual connection
    Matrix residual_output(input_embeddings.rows, input_embeddings.cols);
    for (size_t i = 0; i < residual_output.data.size(); ++i) {
        residual_output.data[i] = input_embeddings.data[i] + context.data[i];
    }

    // Step 6: Layer normalization (simplified - using identity gamma and zero beta)
    Matrix gamma(1, residual_output.cols);
    Matrix beta(1, residual_output.cols);
    std::fill(gamma.data.begin(), gamma.data.end(), 1.0f);
    std::fill(beta.data.begin(), beta.data.end(), 0.0f);
    
    Matrix normalized_output = layer_norm(residual_output, gamma, beta);

    // Step 7: Feed Forward Network
    output = feed_forward(normalized_output, ff_weights1, ff_weights2);
    
    // Final residual connection
    for (size_t i = 0; i < output.data.size(); ++i) {
        output.data[i] += normalized_output.data[i];
    }
}

// Simple data loading functions (since load_binary is not defined)
std::vector<float> load_dummy_data(size_t size) {
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Random values [-1, 1]
    }
    return data;
}

int main() {
    // Initialize random seed
    srand(static_cast<unsigned int>(time(nullptr)));
    
    std::cout << "Starting Transformer Block CPU implementation..." << std::endl;
    
    // Configuration
    int batch_size = 32;    // Reduced for testing
    int seq_length = 64;    // Reduced for testing
    int d_model = 256;      // Reduced for testing
    int d_ff = 1024;        // Feed forward dimension
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length: " << seq_length << std::endl;
    std::cout << "  Model dimension: " << d_model << std::endl;
    std::cout << "  Feed forward dimension: " << d_ff << std::endl;
    
    // Generate dummy data (in practice, these would be loaded from files)
    std::cout << "Generating dummy data..." << std::endl;
    
    auto input_vec = load_dummy_data(batch_size * seq_length * d_model);
    auto weights_q_vec = load_dummy_data(d_model * d_model);
    auto weights_k_vec = load_dummy_data(d_model * d_model);
    auto weights_v_vec = load_dummy_data(d_model * d_model);
    auto ff_weights1_vec = load_dummy_data(d_model * d_ff);
    auto ff_weights2_vec = load_dummy_data(d_ff * d_model);
    
    // Convert to matrices
    Matrix input_embeddings = vector_to_matrix(input_vec, batch_size * seq_length, d_model);
    Matrix weights_q = vector_to_matrix(weights_q_vec, d_model, d_model);
    Matrix weights_k = vector_to_matrix(weights_k_vec, d_model, d_model);
    Matrix weights_v = vector_to_matrix(weights_v_vec, d_model, d_model);
    Matrix ff_weights1 = vector_to_matrix(ff_weights1_vec, d_model, d_ff);
    Matrix ff_weights2 = vector_to_matrix(ff_weights2_vec, d_ff, d_model);
    
    Matrix output(batch_size * seq_length, d_model);
    
    std::cout << "Executing transformer block..." << std::endl;
    
    // Measure execution time
    double execution_time = measureExecutionTime([&]() {
        transformer_block_cpu(input_embeddings, weights_q, weights_k, weights_v, 
                             ff_weights1, ff_weights2, output);
    });
    
    std::cout << "Transformer block execution completed!" << std::endl;
    std::cout << "Execution time: " << execution_time << " seconds" << std::endl;
    
    // Save output
    std::vector<float> output_vec = matrix_to_vector(output);
    saveOutputData("result/cpu_output.bin", output_vec.data(), output_vec.size());
    
    std::cout << "Output saved to result/cpu_output.bin" << std::endl;
    std::cout << "Output matrix dimensions: " << output.rows << " x " << output.cols << std::endl;
    
    return 0;
}