#ifndef SHA256_AVX512_H
#define SHA256_AVX512_H

#include <immintrin.h>
#include <stdint.h>

void sha256avx512_16B(const uint8_t* inputs[16], uint8_t* outputs[16]);

#endif // SHA256_AVX512_H
