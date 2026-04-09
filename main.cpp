#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include "FastLoader.h"
#include "NeonPredictor.h"

void processChunk(const float* data, double* results, const double* weights, 
                  int startRow, int endRow, int cols, double bias) {
    NeonPredictor::predict(&data[startRow * cols], weights, &results[startRow], 
                           (endRow - startRow), cols, bias);
}

int main() {
    FastLoader loader;
    if (!loader.load("test_data.bin")) return 1;

    // Parallel Setup: Optimized for M-series/High-core count silicon
    const int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    double* results = new double[loader.totalRows];

    auto start = std::chrono::high_resolution_clock::now();
    
    int rowsPerThread = loader.totalRows / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i == numThreads - 1) ? loader.totalRows : (i + 1) * rowsPerThread;
        threads.emplace_back(processChunk, loader.data, results, weights, startRow, endRow, 448, 0.0);
    }

    for (auto& t : threads) t.join();

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Throughput: " << loader.totalRows / std::chrono::duration<double>(end - start).count() << " rows/s\n";

    return 0;
}
