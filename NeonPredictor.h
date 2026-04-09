#ifndef NEON_PREDICTOR_H
#define NEON_PREDICTOR_H
#include <arm_neon.h>

class NeonPredictor {
public:
    static void predict(const float* __restrict__ features, 
                        const double* __restrict__ weights,
                        double* __restrict__ output,
                        int num_rows, int num_cols, double bias) {
        
        for (int i = 0; i < num_rows; ++i) {
            float64x2_t sum_vec = vdupq_n_f64(0.0);
            const float* row_feat = &features[i * num_cols];

            for (int j = 0; j <= num_cols - 2; j += 2) {
                // Vectorized Load: 32-bit float to 64-bit double conversion
                float32x2_t f32_vec = vld1_f32(&row_feat[j]);
                float64x2_t f64_vec = vcvt_f64_f32(f32_vec);
                
                float64x2_t w_vec = vld1q_f64(&weights[j]);
                sum_vec = vfmaq_f64(sum_vec, f64_vec, w_vec); // Fused Multiply-Add
            }
            output[i] = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1) + bias;
        }
    }
};
#endif
