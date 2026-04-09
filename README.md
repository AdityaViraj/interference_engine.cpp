# High-Performance HFT Inference Engine

A low-latency C++ inference engine designed for sub-microsecond prediction of financial signals. This project demonstrates the application of **System Architecture** and **OS-level optimizations** to achieve maximum throughput on ARM-based silicon (Apple M-series).

## 🚀 Key System Features

### 1. Zero-Copy Data Ingestion (`mmap`)
Bypassed standard file I/O bottlenecks by implementing **Memory Mapping**. This allows the engine to treat disk-based binary datasets as direct memory arrays, eliminating unnecessary kernel-to-user space copying.

### 2. SIMD Acceleration (ARM NEON)
Leveraged **Single Instruction, Multiple Data (SIMD)** to parallelize the mathematical dot-product.
* **Instruction set:** Used `vfmaq_f64` for Fused Multiply-Add operations.
* **Precision:** Implemented high-precision 64-bit double accumulation for 32-bit float features to ensure "bit-perfect" financial accuracy.

### 3. Hardware-Aware Parallelism
Utilized a multi-threaded execution harness that dynamically scales to the CPU's hardware concurrency, ensuring 100% utilization of available execution units.

### 4. Memory Locality & Numerical Stability
* **Spatial Locality:** Data is stored in row-major order to maximize cache hit rates.
* **Folding Logic:** Implemented mathematical folding of StandardScaler coefficients into the model weights to reduce the number of operations required during live inference.

## 🛠️ Performance
- Throughput: ~24.5M rows/sec (Benchmarked on Apple M4 Pro using 10 hardware threads).
- Latency: ~41 nanoseconds per row (End-to-end inference including SIMD accumulation).

## 📂 Project Structure
- `FastLoader.h`: Memory-mapped file I/O.
- `NeonPredictor.h`: SIMD-accelerated linear regression kernel.
- `train_hft.py`: Robust training pipeline with outlier resistance (Huber Regression).
