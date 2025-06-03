#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "sha256_avx512.h"

namespace _sha256avx512 {

#ifdef _MSC_VER
#define ALIGN64 __declspec(align(64))
#else
#define ALIGN64 __attribute__((aligned(64)))
#endif

// Constants for SHA-256
static const ALIGN64 uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// Initialize SHA-256 state with initial hash values
inline void Initialize(__m512i s[8]) {
  s[0] = _mm512_set1_epi32(0x6a09e667);
  s[1] = _mm512_set1_epi32(0xbb67ae85);
  s[2] = _mm512_set1_epi32(0x3c6ef372);
  s[3] = _mm512_set1_epi32(0xa54ff53a);
  s[4] = _mm512_set1_epi32(0x510e527f);
  s[5] = _mm512_set1_epi32(0x9b05688c);
  s[6] = _mm512_set1_epi32(0x1f83d9ab);
  s[7] = _mm512_set1_epi32(0x5be0cd19);
}

// AVX-512 optimized SHA-256 macros
#define ROR(x, n) _mm512_ror_epi32(x, n)
#define SHR(x, n) _mm512_srli_epi32(x, n)

#define S0(x) _mm512_xor_epi32(ROR(x, 2), _mm512_xor_epi32(ROR(x, 13), ROR(x, 22)))
#define S1(x) _mm512_xor_epi32(ROR(x, 6), _mm512_xor_epi32(ROR(x, 11), ROR(x, 25)))
#define s0(x) _mm512_xor_epi32(ROR(x, 7), _mm512_xor_epi32(ROR(x, 18), SHR(x, 3)))
#define s1(x) _mm512_xor_epi32(ROR(x, 17), _mm512_xor_epi32(ROR(x, 19), SHR(x, 10)))

// Ternary logic operations for improved efficiency
#define Ch(x, y, z) _mm512_ternarylogic_epi32(z, y, x, 0xCA)   // (x & y) ^ (~x & z)
#define Maj(x, y, z) _mm512_ternarylogic_epi32(y, x, z, 0xE8)  // (x & y) ^ (x & z) ^ (y & z)

#define Round(a, b, c, d, e, f, g, h, Kt, Wt)                                                 \
  {                                                                                           \
    __m512i T1 = _mm512_add_epi32(                                                            \
        h, _mm512_add_epi32(S1(e), _mm512_add_epi32(Ch(e, f, g), _mm512_add_epi32(Kt, Wt)))); \
    __m512i T2 = _mm512_add_epi32(S0(a), Maj(a, b, c));                                       \
    h = g;                                                                                    \
    g = f;                                                                                    \
    f = e;                                                                                    \
    e = _mm512_add_epi32(d, T1);                                                              \
    d = c;                                                                                    \
    c = b;                                                                                    \
    b = a;                                                                                    \
    a = _mm512_add_epi32(T1, T2);                                                             \
  }

inline void Transform(__m512i* state, const uint8_t* data[16]) {
  __m512i a = state[0], b = state[1], c = state[2], d = state[3];
  __m512i e = state[4], f = state[5], g = state[6], h = state[7];
  __m512i W[64];

  // Prepare message schedule W[0..15] using direct memory access
  for (int t = 0; t < 16; t++) {
    ALIGN64 uint32_t wt[16];
    for (int j = 0; j < 16; j++) {
      const uint8_t* ptr = data[j] + t * 4;
      wt[j] =
          ((uint32_t)ptr[0] << 24) | ((uint32_t)ptr[1] << 16) | ((uint32_t)ptr[2] << 8) | ptr[3];
    }
    W[t] = _mm512_load_epi32(wt);
  }

  // Main loop of SHA-256 with AVX-512 optimizations
  for (int t = 0; t < 16; ++t) {
    Round(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[t]), W[t]);
  }

  for (int t = 16; t < 64; ++t) {
    __m512i newW = _mm512_add_epi32(_mm512_add_epi32(s1(W[t - 2]), W[t - 7]),
                                    _mm512_add_epi32(s0(W[t - 15]), W[t - 16]));
    W[t] = newW;
    Round(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[t]), newW);
  }

  // Update state
  state[0] = _mm512_add_epi32(state[0], a);
  state[1] = _mm512_add_epi32(state[1], b);
  state[2] = _mm512_add_epi32(state[2], c);
  state[3] = _mm512_add_epi32(state[3], d);
  state[4] = _mm512_add_epi32(state[4], e);
  state[5] = _mm512_add_epi32(state[5], f);
  state[6] = _mm512_add_epi32(state[6], g);
  state[7] = _mm512_add_epi32(state[7], h);
}

}  // namespace _sha256avx512

void sha256_avx512_16B(const uint8_t* data[16], unsigned char* hash[16]) {
  ALIGN64 __m512i state[8];

  // Initialize the state
  _sha256avx512::Initialize(state);

  // Process the data blocks
  _sha256avx512::Transform(state, data);

  // Store the resulting state
  ALIGN64 uint32_t digest[8][16];  // [state_index][element_index]

  // Extract and store hash values with byte swapping
  for (int i = 0; i < 8; ++i) {
    _mm512_store_epi32(digest[i], state[i]);
  }

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 8; ++j) {
      uint32_t word = digest[j][i];
      word = __builtin_bswap32(word);
      memcpy(hash[i] + j * 4, &word, 4);
    }
  }
}