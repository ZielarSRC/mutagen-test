#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "sha256_avx512.h"

// SHA-256 constants
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

void sha256avx512_16B(const uint8_t* inputs[16], uint8_t* outputs[16]) {
  alignas(64) uint32_t W[64][16];

  // Message schedule preparation: load and transpose input blocks
  for (int blk = 0; blk < 16; ++blk) {
    for (int t = 0; t < 16; ++t) {
      // Load 4 bytes as big-endian word
      W[t][blk] = ((uint32_t)inputs[blk][t * 4 + 0] << 24) |
                  ((uint32_t)inputs[blk][t * 4 + 1] << 16) |
                  ((uint32_t)inputs[blk][t * 4 + 2] << 8) | ((uint32_t)inputs[blk][t * 4 + 3]);
    }
  }

  // Extend the first 16 words into the remaining 48 words
  for (int t = 16; t < 64; ++t) {
    for (int blk = 0; blk < 16; ++blk) {
      uint32_t w15 = W[t - 15][blk];
      uint32_t w2 = W[t - 2][blk];
      uint32_t S0v =
          ((w15 >> 7) | (w15 << (32 - 7))) ^ ((w15 >> 18) | (w15 << (32 - 18))) ^ (w15 >> 3);
      uint32_t S1v =
          ((w2 >> 17) | (w2 << (32 - 17))) ^ ((w2 >> 19) | (w2 << (32 - 19))) ^ (w2 >> 10);
      W[t][blk] = W[t - 16][blk] + S0v + W[t - 7][blk] + S1v;
    }
  }

  // Initialize hash value for each lane
  __m512i a = _mm512_set1_epi32(0x6A09E667);
  __m512i b = _mm512_set1_epi32(0xBB67AE85);
  __m512i c = _mm512_set1_epi32(0x3C6EF372);
  __m512i d = _mm512_set1_epi32(0xA54FF53A);
  __m512i e = _mm512_set1_epi32(0x510E527F);
  __m512i f = _mm512_set1_epi32(0x9B05688C);
  __m512i g = _mm512_set1_epi32(0x1F83D9AB);
  __m512i h = _mm512_set1_epi32(0x5BE0CD19);

  // Load initial state for each lane
  for (int lane = 0; lane < 16; ++lane) {
    ((uint32_t*)&a)[lane] = 0x6A09E667;
    ((uint32_t*)&b)[lane] = 0xBB67AE85;
    ((uint32_t*)&c)[lane] = 0x3C6EF372;
    ((uint32_t*)&d)[lane] = 0xA54FF53A;
    ((uint32_t*)&e)[lane] = 0x510E527F;
    ((uint32_t*)&f)[lane] = 0x9B05688C;
    ((uint32_t*)&g)[lane] = 0x1F83D9AB;
    ((uint32_t*)&h)[lane] = 0x5BE0CD19;
  }

  // Main compression loop
  for (int t = 0; t < 64; ++t) {
    // Gather W[t][blk] into one vector (lane 0 = blk 0, lane 15 = blk 15)
    __m512i Wt = _mm512_set_epi32(W[t][15], W[t][14], W[t][13], W[t][12], W[t][11], W[t][10],
                                  W[t][9], W[t][8], W[t][7], W[t][6], W[t][5], W[t][4], W[t][3],
                                  W[t][2], W[t][1], W[t][0]);
    __m512i Kt = _mm512_set1_epi32(K[t]);
    __m512i T1 = _mm512_add_epi32(
        h, _mm512_add_epi32(S1(e), _mm512_add_epi32(Ch(e, f, g), _mm512_add_epi32(Kt, Wt))));
    __m512i T2 = _mm512_add_epi32(S0(a), Maj(a, b, c));
    h = g;
    g = f;
    f = e;
    e = _mm512_add_epi32(d, T1);
    d = c;
    c = b;
    b = a;
    a = _mm512_add_epi32(T1, T2);
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

  // Store results lane-wise: outputs[i] = hash for inputs[i]
  alignas(64) uint32_t result[8][16];
  _mm512_store_epi32(result[0], a);
  _mm512_store_epi32(result[1], b);
  _mm512_store_epi32(result[2], c);
  _mm512_store_epi32(result[3], d);
  _mm512_store_epi32(result[4], e);
  _mm512_store_epi32(result[5], f);
  _mm512_store_epi32(result[6], g);
  _mm512_store_epi32(result[7], h);

  for (int blk = 0; blk < 16; ++blk) {
    for (int word = 0; word < 8; ++word) {
      uint32_t w = result[word][blk];
      outputs[blk][word * 4 + 0] = (w >> 24) & 0xff;
      outputs[blk][word * 4 + 1] = (w >> 16) & 0xff;
      outputs[blk][word * 4 + 2] = (w >> 8) & 0xff;
      outputs[blk][word * 4 + 3] = (w >> 0) & 0xff;
    }
  }
}
