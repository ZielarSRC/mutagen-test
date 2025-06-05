#include <immintrin.h>

#include <cstring>

#include "sha256_avx512.h"

static const uint32_t K[64] = {
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2};

#define ROTR32(x, n) _mm512_ror_epi32(x, n)
#define SHR32(x, n) _mm512_srli_epi32(x, n)

#define S0(x) (_mm512_xor_si512(_mm512_xor_si512(ROTR32(x, 2), ROTR32(x, 13)), ROTR32(x, 22)))
#define S1(x) (_mm512_xor_si512(_mm512_xor_si512(ROTR32(x, 6), ROTR32(x, 11)), ROTR32(x, 25)))
#define s0(x) (_mm512_xor_si512(_mm512_xor_si512(ROTR32(x, 7), ROTR32(x, 18)), SHR32(x, 3)))
#define s1(x) (_mm512_xor_si512(_mm512_xor_si512(ROTR32(x, 17), ROTR32(x, 19)), SHR32(x, 10)))

#define Ch(x, y, z) _mm512_ternarylogic_epi32(z, y, x, 0xCA)
#define Maj(x, y, z) _mm512_ternarylogic_epi32(y, x, z, 0xE8)

#define Round(a, b, c, d, e, f, g, h, Kt, Wt)                                                 \
  {                                                                                           \
    __m512i T1 = _mm512_add_epi32(                                                            \
        h, _mm512_add_epi32(S1(e), _mm512_add_epi32(Ch(e, f, g), _mm512_add_epi32(Kt, Wt)))); \
    __m512i T2 = _mm512_add_epi32(S0(a), Maj(a, b, c));                                       \
    d = _mm512_add_epi32(d, T1);                                                              \
    h = _mm512_add_epi32(T1, T2);                                                             \
  }

void sha256avx512_16B(const uint8_t* inputs[16], uint8_t* outputs[16]) {
  alignas(64) uint32_t message_schedule[16][16];

  // Load and transpose input data
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      message_schedule[j][i] = __builtin_bswap32(((uint32_t*)inputs[i])[j]);
    }
  }

  __m512i W[64];

  // Load first 16 words
  for (int i = 0; i < 16; i++) {
    W[i] = _mm512_load_epi32(message_schedule[i]);
  }

  // Extend the first 16 words into the remaining 48 words
  for (int i = 16; i < 64; i++) {
    W[i] = _mm512_add_epi32(
        _mm512_add_epi32(_mm512_add_epi32(s1(W[i - 2]), W[i - 7]), s0(W[i - 15])), W[i - 16]);
  }

  // Initialize hash values
  __m512i a = _mm512_set1_epi32(0x6A09E667);
  __m512i b = _mm512_set1_epi32(0xBB67AE85);
  __m512i c = _mm512_set1_epi32(0x3C6EF372);
  __m512i d = _mm512_set1_epi32(0xA54FF53A);
  __m512i e = _mm512_set1_epi32(0x510E527F);
  __m512i f = _mm512_set1_epi32(0x9B05688C);
  __m512i g = _mm512_set1_epi32(0x1F83D9AB);
  __m512i h = _mm512_set1_epi32(0x5BE0CD19);

  // Main compression loop
  for (int i = 0; i < 64; i++) {
    __m512i Kt = _mm512_set1_epi32(K[i]);
    Round(a, b, c, d, e, f, g, h, Kt, W[i]);

    // Rotate variables
    __m512i temp = h;
    h = g;
    g = f;
    f = e;
    e = d;
    d = c;
    c = b;
    b = a;
    a = temp;
  }

  // Add the compressed chunk to the current hash value
  a = _mm512_add_epi32(a, _mm512_set1_epi32(0x6A09E667));
  b = _mm512_add_epi32(b, _mm512_set1_epi32(0xBB67AE85));
  c = _mm512_add_epi32(c, _mm512_set1_epi32(0x3C6EF372));
  d = _mm512_add_epi32(d, _mm512_set1_epi32(0xA54FF53A));
  e = _mm512_add_epi32(e, _mm512_set1_epi32(0x510E527F));
  f = _mm512_add_epi32(f, _mm512_set1_epi32(0x9B05688C));
  g = _mm512_add_epi32(g, _mm512_set1_epi32(0x1F83D9AB));
  h = _mm512_add_epi32(h, _mm512_set1_epi32(0x5BE0CD19));

  // Store results
  alignas(64) uint32_t result[8][16];
  _mm512_store_epi32(result[0], a);
  _mm512_store_epi32(result[1], b);
  _mm512_store_epi32(result[2], c);
  _mm512_store_epi32(result[3], d);
  _mm512_store_epi32(result[4], e);
  _mm512_store_epi32(result[5], f);
  _mm512_store_epi32(result[6], g);
  _mm512_store_epi32(result[7], h);

  // Transpose and store output
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 8; j++) {
      ((uint32_t*)outputs[i])[j] = __builtin_bswap32(result[j][i]);
    }
  }
}
