#ifndef SHA256_AVX512_H
#define SHA256_AVX512_H

#include <stdint.h>

void sha256_avx512_16B(const uint8_t* data[16], unsigned char* hash[16]);

#endif  // SHA256_AVX512_H