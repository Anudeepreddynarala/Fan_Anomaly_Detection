// ESP-DSP engine integration - DISABLED FOR PORTABLE BUILD
#ifndef EI_ESP_DSP_H
#define EI_ESP_DSP_H

#include <stdint.h>
#include <stdbool.h>

namespace ei {
namespace fft {

// Constants
constexpr int MIN_FFT_SIZE = 4;
constexpr int MAX_FFT_SIZE = 4096;

// All functions stubbed to return false/error (ESP-DSP not available)
static inline bool can_do_fft(size_t n_fft) {
    (void)n_fft;
    return false;
}

static inline bool init_fft(size_t n_fft) {
    (void)n_fft;
    return false;
}

static inline void deinit_fft() {
}

static inline int fft(float *input, size_t n_fft) {
    (void)input;
    (void)n_fft;
    return -1;
}

template<typename T>
static inline int hw_r2c_fft(float *input, T *output, size_t n_fft) {
    (void)input;
    (void)output;
    (void)n_fft;
    return -1; // Not available
}

} // namespace fft
} // namespace ei

#endif // EI_ESP_DSP_H
