// knn_cpu.cpp
// Single‑threaded reference for squared‑L2 k‑NN.
// Build for a CLI test tool:   g++ -O3 -std=c++17 -DBUILD_STANDALONE knn_cpu.cpp -o knn_cpu
// Build as a library (omit main)   g++ -O3 -std=c++17 -c knn_cpu.cpp

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// -----------------------------------------------------------------------------
// Small, allocation‑free helper doing exhaustive k‑NN on the CPU.
//   database : N × d  row‑major
//   queries  : Q × d  row‑major
//   indices_out  : Q × k
//   distances_out: Q × k   (squared‑L2)
// -----------------------------------------------------------------------------
void knn_cpu(const float* database,
             const float* queries,
             int32_t*     indices_out,
             float*       distances_out,
             int N, int Q, int d, int k)
{
    // Pre‑compute ‖d‖² once -> saves N*Q multiplies.
    std::vector<float> db_norm2(N);
    for (int i = 0; i < N; ++i) {
        float s = 0.f;
        for (int j = 0; j < d; ++j) {
            float v = database[i * d + j];
            s += v * v;
        }
        db_norm2[i] = s;
    }

    struct Pair { float dist; int idx; };

    for (int q = 0; q < Q; ++q) {
        const float* qvec = queries + q * d;

        float q_norm2 = 0.f;
        for (int j = 0; j < d; ++j) q_norm2 += qvec[j] * qvec[j];

        std::vector<Pair> buf(N);
        for (int i = 0; i < N; ++i) {
            const float* dvec = database + i * d;
            float dot = 0.f;
            for (int j = 0; j < d; ++j) dot += qvec[j] * dvec[j];
            float dist = q_norm2 + db_norm2[i] - 2.f * dot;
            buf[i] = {dist, i};
        }

        std::partial_sort(buf.begin(), buf.begin() + k, buf.end(),
                          [](const Pair& a, const Pair& b){ return a.dist < b.dist; });

        for (int r = 0; r < k; ++r) {
            distances_out[q * k + r] = buf[r].dist;
            indices_out  [q * k + r] = buf[r].idx;
        }
    }
}

// -----------------------------------------------------------------------------
// Optional CLI harness so you can just run  ./knn_cpu 100000 1024 128 10
// -----------------------------------------------------------------------------
#ifdef BUILD_STANDALONE
#include <chrono>

int main(int argc, char** argv)
{
    if (argc != 5) {
        std::printf("Usage: %s N Q d k\n", argv[0]);
        return 0;
    }
    int N = std::atoi(argv[1]);
    int Q = std::atoi(argv[2]);
    int d = std::atoi(argv[3]);
    int k = std::atoi(argv[4]);

    std::vector<float> database(static_cast<size_t>(N) * d);
    std::vector<float> queries (static_cast<size_t>(Q) * d);

    // Deterministic pseudo‑random values (good enough for a demo)
    for (size_t i = 0; i < database.size(); ++i) database[i] = float((i * 37) % 1000) / 1000.f;
    for (size_t i = 0; i < queries .size(); ++i) queries [i] = float((i * 17) % 1000) / 1000.f;

    std::vector<int32_t> indices (static_cast<size_t>(Q) * k);
    std::vector<float>   distances(static_cast<size_t>(Q) * k);

    auto t0 = std::chrono::high_resolution_clock::now();
    knn_cpu(database.data(), queries.data(),
            indices.data(), distances.data(), N, Q, d, k);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("CPU k‑NN finished in %.3f ms  (N=%d Q=%d d=%d k=%d)\n",
                ms, N, Q, d, k);
    return 0;
}
#endif
