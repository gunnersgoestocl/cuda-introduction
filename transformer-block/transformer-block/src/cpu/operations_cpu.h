#ifndef OPERATIONS_CPU_H
#define OPERATIONS_CPU_H

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

// Function declarations for CPU operations
MatrixXf matmul(const MatrixXf& A, const MatrixXf& B);
MatrixXf softmax(const MatrixXf& scores);
MatrixXf layer_norm(const MatrixXf& input, const MatrixXf& gamma, const MatrixXf& beta);
MatrixXf feed_forward(const MatrixXf& input, const MatrixXf& weights1, const MatrixXf& weights2);

#endif // OPERATIONS_CPU_H