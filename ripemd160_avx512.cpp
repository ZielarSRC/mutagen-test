#include <immintrin.h>

#include <cstdint>
#include <cstring>

#include "ripemd160_avx512.h"

namespace ripemd160avx512 {

#ifdef WIN64
static const __declspec(align(64)) uint32_t _init[] = {
#else
static const uint32_t _init[] __attribute__((aligned(64))) = {
#endif
    // 16 copies of A
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,

    // 16 copies of B
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,

    // 16 copies of C
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,

    // 16 copies of D
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,

    // 16 copies of E
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul,
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul,
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul};

// AVX-512 optimized operations
#define ROL(x, n) _mm512_rol_epi32(x, n)

// RIPEMD-160 functions using AVX-512 ternary logic
#define f1(x, y, z) _mm512_xor_si512(x, _mm512_xor_si512(y, z))
#define f2(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0xCA)  // (x & y) | (~x & z)
#define f3(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0x96)  // (x | ~y) ^ z
#define f4(x, y, z) _mm512_ternarylogic_epi32(x, z, y, 0xAC)  // (x & z) | (y & ~z)
#define f5(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0x3C)  // x ^ (y | ~z)

// Efficient addition helpers
#define add3(x0, x1, x2) _mm512_add_epi32(_mm512_add_epi32(x0, x1), x2)
#define add4(x0, x1, x2, x3) _mm512_add_epi32(_mm512_add_epi32(x0, x1), _mm512_add_epi32(x2, x3))

// Round macro with local variable declaration
#define Round(a, b, c, d, e, f, x, k, r)             \
  do {                                               \
    __m512i u = add4(a, f, x, _mm512_set1_epi32(k)); \
    a = _mm512_add_epi32(ROL(u, r), e);              \
    c = ROL(c, 10);                                  \
  } while (0)

// Round macros with proper function usage
#define R11(a, b, c, d, e, x, r) Round(a, b, c, d, e, f1(b, c, d), x, 0, r)
#define R21(a, b, c, d, e, x, r) Round(a, b, c, d, e, f2(b, c, d), x, 0x5A827999ul, r)
#define R31(a, b, c, d, e, x, r) Round(a, b, c, d, e, f3(b, c, d), x, 0x6ED9EBA1ul, r)
#define R41(a, b, c, d, e, x, r) Round(a, b, c, d, e, f4(b, c, d), x, 0x8F1BBCDCul, r)
#define R51(a, b, c, d, e, x, r) Round(a, b, c, d, e, f5(b, c, d), x, 0xA953FD4Eul, r)
#define R12(a, b, c, d, e, x, r) Round(a, b, c, d, e, f5(b, c, d), x, 0x50A28BE6ul, r)
#define R22(a, b, c, d, e, x, r) Round(a, b, c, d, e, f4(b, c, d), x, 0x5C4DD124ul, r)
#define R32(a, b, c, d, e, x, r) Round(a, b, c, d, e, f3(b, c, d), x, 0x6D703EF3ul, r)
#define R42(a, b, c, d, e, x, r) Round(a, b, c, d, e, f2(b, c, d), x, 0x7A6D76E9ul, r)
#define R52(a, b, c, d, e, x, r) Round(a, b, c, d, e, f1(b, c, d), x, 0, r)

// Load message data with proper byte order handling
#define LOADW(i)                                                                             \
  _mm512_set_epi32(                                                                          \
      ((uint32_t *)inputs[15])[i], ((uint32_t *)inputs[14])[i], ((uint32_t *)inputs[13])[i], \
      ((uint32_t *)inputs[12])[i], ((uint32_t *)inputs[11])[i], ((uint32_t *)inputs[10])[i], \
      ((uint32_t *)inputs[9])[i], ((uint32_t *)inputs[8])[i], ((uint32_t *)inputs[7])[i],    \
      ((uint32_t *)inputs[6])[i], ((uint32_t *)inputs[5])[i], ((uint32_t *)inputs[4])[i],    \
      ((uint32_t *)inputs[3])[i], ((uint32_t *)inputs[2])[i], ((uint32_t *)inputs[1])[i],    \
      ((uint32_t *)inputs[0])[i])

// Initialize RIPEMD-160 state
void Initialize(__m512i state[5]) { memcpy(state, _init, sizeof(_init)); }

// Process 16 message blocks simultaneously
void Transform(__m512i state[5], const uint8_t *inputs[16]) {
  __m512i a1 = state[0], b1 = state[1], c1 = state[2], d1 = state[3], e1 = state[4];
  __m512i a2 = a1, b2 = b1, c2 = c1, d2 = d1, e2 = e1;
  __m512i w[16];

  // Load all 16 message words
  for (int i = 0; i < 16; ++i) {
    w[i] = LOADW(i);
  }

  // Round 1 - Left path and right path interleaved
  R11(a1, b1, c1, d1, e1, w[0], 11);
  R12(a2, b2, c2, d2, e2, w[5], 8);
  R11(e1, a1, b1, c1, d1, w[1], 14);
  R12(e2, a2, b2, c2, d2, w[14], 9);
  R11(d1, e1, a1, b1, c1, w[2], 15);
  R12(d2, e2, a2, b2, c2, w[7], 9);
  R11(c1, d1, e1, a1, b1, w[3], 12);
  R12(c2, d2, e2, a2, b2, w[0], 11);
  R11(b1, c1, d1, e1, a1, w[4], 5);
  R12(b2, c2, d2, e2, a2, w[9], 13);
  R11(a1, b1, c1, d1, e1, w[5], 8);
  R12(a2, b2, c2, d2, e2, w[2], 15);
  R11(e1, a1, b1, c1, d1, w[6], 7);
  R12(e2, a2, b2, c2, d2, w[11], 15);
  R11(d1, e1, a1, b1, c1, w[7], 9);
  R12(d2, e2, a2, b2, c2, w[4], 5);
  R11(c1, d1, e1, a1, b1, w[8], 11);
  R12(c2, d2, e2, a2, b2, w[13], 7);
  R11(b1, c1, d1, e1, a1, w[9], 13);
  R12(b2, c2, d2, e2, a2, w[6], 7);
  R11(a1, b1, c1, d1, e1, w[10], 14);
  R12(a2, b2, c2, d2, e2, w[15], 8);
  R11(e1, a1, b1, c1, d1, w[11], 15);
  R12(e2, a2, b2, c2, d2, w[8], 11);
  R11(d1, e1, a1, b1, c1, w[12], 6);
  R12(d2, e2, a2, b2, c2, w[1], 14);
  R11(c1, d1, e1, a1, b1, w[13], 7);
  R12(c2, d2, e2, a2, b2, w[10], 14);
  R11(b1, c1, d1, e1, a1, w[14], 9);
  R12(b2, c2, d2, e2, a2, w[3], 12);
  R11(a1, b1, c1, d1, e1, w[15], 8);
  R12(a2, b2, c2, d2, e2, w[12], 6);

  // Round 2
  R21(e1, a1, b1, c1, d1, w[7], 7);
  R22(e2, a2, b2, c2, d2, w[6], 9);
  R21(d1, e1, a1, b1, c1, w[4], 6);
  R22(d2, e2, a2, b2, c2, w[11], 13);
  R21(c1, d1, e1, a1, b1, w[13], 8);
  R22(c2, d2, e2, a2, b2, w[3], 15);
  R21(b1, c1, d1, e1, a1, w[1], 13);
  R22(b2, c2, d2, e2, a2, w[7], 7);
  R21(a1, b1, c1, d1, e1, w[10], 11);
  R22(a2, b2, c2, d2, e2, w[0], 12);
  R21(e1, a1, b1, c1, d1, w[6], 9);
  R22(e2, a2, b2, c2, d2, w[13], 8);
  R21(d1, e1, a1, b1, c1, w[15], 7);
  R22(d2, e2, a2, b2, c2, w[5], 9);
  R21(c1, d1, e1, a1, b1, w[3], 15);
  R22(c2, d2, e2, a2, b2, w[10], 11);
  R21(b1, c1, d1, e1, a1, w[12], 7);
  R22(b2, c2, d2, e2, a2, w[14], 7);
  R21(a1, b1, c1, d1, e1, w[0], 12);
  R22(a2, b2, c2, d2, e2, w[15], 7);
  R21(e1, a1, b1, c1, d1, w[9], 15);
  R22(e2, a2, b2, c2, d2, w[8], 12);
  R21(d1, e1, a1, b1, c1, w[5], 9);
  R22(d2, e2, a2, b2, c2, w[12], 7);
  R21(c1, d1, e1, a1, b1, w[2], 11);
  R22(c2, d2, e2, a2, b2, w[4], 6);
  R21(b1, c1, d1, e1, a1, w[14], 7);
  R22(b2, c2, d2, e2, a2, w[9], 15);
  R21(a1, b1, c1, d1, e1, w[11], 13);
  R22(a2, b2, c2, d2, e2, w[1], 13);
  R21(e1, a1, b1, c1, d1, w[8], 12);
  R22(e2, a2, b2, c2, d2, w[2], 11);

  // Round 3
  R31(d1, e1, a1, b1, c1, w[3], 11);
  R32(d2, e2, a2, b2, c2, w[15], 9);
  R31(c1, d1, e1, a1, b1, w[10], 13);
  R32(c2, d2, e2, a2, b2, w[5], 7);
  R31(b1, c1, d1, e1, a1, w[14], 6);
  R32(b2, c2, d2, e2, a2, w[1], 15);
  R31(a1, b1, c1, d1, e1, w[4], 7);
  R32(a2, b2, c2, d2, e2, w[3], 11);
  R31(e1, a1, b1, c1, d1, w[9], 14);
  R32(e2, a2, b2, c2, d2, w[7], 8);
  R31(d1, e1, a1, b1, c1, w[15], 9);
  R32(d2, e2, a2, b2, c2, w[14], 6);
  R31(c1, d1, e1, a1, b1, w[8], 13);
  R32(c2, d2, e2, a2, b2, w[6], 6);
  R31(b1, c1, d1, e1, a1, w[1], 15);
  R32(b2, c2, d2, e2, a2, w[9], 14);
  R31(a1, b1, c1, d1, e1, w[2], 14);
  R32(a2, b2, c2, d2, e2, w[11], 12);
  R31(e1, a1, b1, c1, d1, w[7], 8);
  R32(e2, a2, b2, c2, d2, w[8], 13);
  R31(d1, e1, a1, b1, c1, w[0], 13);
  R32(d2, e2, a2, b2, c2, w[12], 5);
  R31(c1, d1, e1, a1, b1, w[6], 6);
  R32(c2, d2, e2, a2, b2, w[2], 14);
  R31(b1, c1, d1, e1, a1, w[13], 5);
  R32(b2, c2, d2, e2, a2, w[10], 13);
  R31(a1, b1, c1, d1, e1, w[11], 12);
  R32(a2, b2, c2, d2, e2, w[0], 13);
  R31(e1, a1, b1, c1, d1, w[5], 7);
  R32(e2, a2, b2, c2, d2, w[4], 7);
  R31(d1, e1, a1, b1, c1, w[12], 5);
  R32(d2, e2, a2, b2, c2, w[13], 5);

  // Round 4
  R41(c1, d1, e1, a1, b1, w[1], 11);
  R42(c2, d2, e2, a2, b2, w[8], 15);
  R41(b1, c1, d1, e1, a1, w[9], 12);
  R42(b2, c2, d2, e2, a2, w[6], 5);
  R41(a1, b1, c1, d1, e1, w[11], 14);
  R42(a2, b2, c2, d2, e2, w[4], 8);
  R41(e1, a1, b1, c1, d1, w[10], 15);
  R42(e2, a2, b2, c2, d2, w[1], 11);
  R41(d1, e1, a1, b1, c1, w[0], 14);
  R42(d2, e2, a2, b2, c2, w[3], 14);
  R41(c1, d1, e1, a1, b1, w[8], 15);
  R42(c2, d2, e2, a2, b2, w[11], 14);
  R41(b1, c1, d1, e1, a1, w[12], 9);
  R42(b2, c2, d2, e2, a2, w[15], 6);
  R41(a1, b1, c1, d1, e1, w[4], 8);
  R42(a2, b2, c2, d2, e2, w[0], 14);
  R41(e1, a1, b1, c1, d1, w[13], 9);
  R42(e2, a2, b2, c2, d2, w[5], 6);
  R41(d1, e1, a1, b1, c1, w[3], 14);
  R42(d2, e2, a2, b2, c2, w[12], 9);
  R41(c1, d1, e1, a1, b1, w[7], 5);
  R42(c2, d2, e2, a2, b2, w[2], 12);
  R41(b1, c1, d1, e1, a1, w[15], 6);
  R42(b2, c2, d2, e2, a2, w[13], 9);
  R41(a1, b1, c1, d1, e1, w[14], 8);
  R42(a2, b2, c2, d2, e2, w[9], 12);
  R41(e1, a1, b1, c1, d1, w[5], 6);
  R42(e2, a2, b2, c2, d2, w[7], 5);
  R41(d1, e1, a1, b1, c1, w[6], 5);
  R42(d2, e2, a2, b2, c2, w[10], 15);
  R41(c1, d1, e1, a1, b1, w[2], 12);
  R42(c2, d2, e2, a2, b2, w[14], 8);

  // Round 5
  R51(b1, c1, d1, e1, a1, w[4], 9);
  R52(b2, c2, d2, e2, a2, w[12], 8);
  R51(a1, b1, c1, d1, e1, w[0], 15);
  R52(a2, b2, c2, d2, e2, w[15], 5);
  R51(e1, a1, b1, c1, d1, w[5], 5);
  R52(e2, a2, b2, c2, d2, w[10], 12);
  R51(d1, e1, a1, b1, c1, w[9], 11);
  R52(d2, e2, a2, b2, c2, w[4], 9);
  R51(c1, d1, e1, a1, b1, w[7], 6);
  R52(c2, d2, e2, a2, b2, w[1], 12);
  R51(b1, c1, d1, e1, a1, w[12], 8);
  R52(b2, c2, d2, e2, a2, w[5], 5);
  R51(a1, b1, c1, d1, e1, w[2], 13);
  R52(a2, b2, c2, d2, e2, w[8], 14);
  R51(e1, a1, b1, c1, d1, w[10], 12);
  R52(e2, a2, b2, c2, d2, w[7], 6);
  R51(d1, e1, a1, b1, c1, w[14], 5);
  R52(d2, e2, a2, b2, c2, w[6], 8);
  R51(c1, d1, e1, a1, b1, w[1], 12);
  R52(c2, d2, e2, a2, b2, w[2], 13);
  R51(b1, c1, d1, e1, a1, w[3], 13);
  R52(b2, c2, d2, e2, a2, w[13], 6);
  R51(a1, b1, c1, d1, e1, w[8], 14);
  R52(a2, b2, c2, d2, e2, w[14], 5);
  R51(e1, a1, b1, c1, d1, w[11], 11);
  R52(e2, a2, b2, c2, d2, w[0], 15);
  R51(d1, e1, a1, b1, c1, w[6], 8);
  R52(d2, e2, a2, b2, c2, w[3], 13);
  R51(c1, d1, e1, a1, b1, w[15], 5);
  R52(c2, d2, e2, a2, b2, w[9], 11);
  R51(b1, c1, d1, e1, a1, w[13], 6);
  R52(b2, c2, d2, e2, a2, w[11], 11);

  // Combine results following RIPEMD-160 specification
  __m512i t = state[0];
  state[0] = add3(state[1], c1, d2);
  state[1] = add3(state[2], d1, e2);
  state[2] = add3(state[3], e1, a2);
  state[3] = add3(state[4], a1, b2);
  state[4] = add3(t, b1, c2);
}

// Result extraction macro with proper indexing
#define DEPACK(outputs, idx)                      \
  do {                                            \
    uint32_t *s0 = (uint32_t *)&state[0];         \
    uint32_t *s1 = (uint32_t *)&state[1];         \
    uint32_t *s2 = (uint32_t *)&state[2];         \
    uint32_t *s3 = (uint32_t *)&state[3];         \
    uint32_t *s4 = (uint32_t *)&state[4];         \
    ((uint32_t *)outputs[idx])[0] = s0[15 - idx]; \
    ((uint32_t *)outputs[idx])[1] = s1[15 - idx]; \
    ((uint32_t *)outputs[idx])[2] = s2[15 - idx]; \
    ((uint32_t *)outputs[idx])[3] = s3[15 - idx]; \
    ((uint32_t *)outputs[idx])[4] = s4[15 - idx]; \
  } while (0)

// Main RIPEMD-160 hashing function compatible with original interface
void ripemd160avx512_16(const uint8_t *inputs[16], uint8_t *outputs[16]) {
  __m512i state[5];

  // Initialize RIPEMD-160 state
  Initialize(state);

  // Process all blocks
  Transform(state, inputs);

  // Extract results for each input block
  for (int i = 0; i < 16; i++) {
    DEPACK(outputs, i);
  }
}

}  // namespace ripemd160avx512
