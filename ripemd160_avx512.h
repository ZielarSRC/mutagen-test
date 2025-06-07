#ifndef RIPEMD160_AVX512_H
#define RIPEMD160_AVX512_H

#include <immintrin.h>
#include <stdint.h>

namespace ripemd160avx512 {

// Processes 16 blocks of 32 bytes each.
// Each outputs[i] receives a 20-byte RIPEMD-160 hash for inputs[i].
void ripemd160avx512_16(const uint8_t* inputs[16], uint8_t* outputs[16]);

}  // namespace ripemd160avx512

#endif  // RIPEMD160_AVX512_H
