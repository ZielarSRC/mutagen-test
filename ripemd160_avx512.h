#ifndef RIPEMD160_AVX512_H
#define RIPEMD160_AVX512_H

#include <immintrin.h>
#include <stdint.h>

namespace ripemd160avx512 {
void ripemd160avx512_16(const uint8_t* inputs[16], uint8_t* outputs[16]);
}

#endif  // RIPEMD160_AVX512_H
