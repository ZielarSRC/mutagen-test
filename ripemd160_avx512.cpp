#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "ripemd160_avx512.h"

namespace ripemd160avx512 {

// Helper for little-endian load
static inline uint32_t read_le32(const uint8_t* p) {
  return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Perform 16-way RIPEMD-160 for 32-byte inputs, store 20-byte outputs per input
void ripemd160avx512_16(const uint8_t* inputs[16], uint8_t* outputs[16]) {
  static const uint32_t h0_init = 0x67452301UL;
  static const uint32_t h1_init = 0xEFCDAB89UL;
  static const uint32_t h2_init = 0x98BADCFEUL;
  static const uint32_t h3_init = 0x10325476UL;
  static const uint32_t h4_init = 0xC3D2E1F0UL;

  static const uint32_t K[5] = {0x00000000UL, 0x5A827999UL, 0x6ED9EBA1UL, 0x8F1BBCDCUL,
                                0xA953FD4EUL};
  static const uint32_t KK[5] = {0x50A28BE6UL, 0x5C4DD124UL, 0x6D703EF3UL, 0x7A6D76E9UL,
                                 0x00000000UL};

  static const uint8_t RL[80] = {11, 14, 15, 12, 5,  8,  7,  9,  11, 13, 14, 15, 6,  7,  9,  8,
                                 7,  6,  8,  13, 11, 9,  7,  15, 7,  12, 15, 9,  11, 7,  13, 12,
                                 11, 13, 6,  7,  14, 9,  13, 15, 14, 8,  13, 6,  5,  12, 7,  5,
                                 11, 12, 14, 15, 14, 15, 9,  8,  9,  14, 5,  6,  8,  6,  5,  12,
                                 9,  15, 5,  11, 6,  8,  13, 12, 5,  12, 13, 14, 11, 8,  5,  6};
  static const uint8_t RR[80] = {8,  9,  9,  11, 13, 15, 15, 5,  7,  7,  8,  11, 14, 14, 12, 6,
                                 9,  13, 15, 7,  12, 8,  9,  11, 7,  7,  12, 7,  6,  15, 13, 11,
                                 9,  7,  15, 11, 8,  6,  6,  14, 12, 13, 5,  14, 13, 13, 7,  5,
                                 15, 5,  8,  11, 14, 14, 6,  14, 6,  9,  12, 9,  12, 5,  15, 8,
                                 8,  5,  12, 9,  12, 5,  14, 6,  8,  13, 6,  5,  15, 13, 11, 11};
  static const uint8_t SL[80] = {0, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
                                 7, 4,  13, 1,  10, 6,  15, 3,  12, 0, 9,  5,  2,  14, 11, 8,
                                 3, 10, 14, 4,  9,  15, 8,  1,  2,  7, 0,  6,  13, 11, 5,  12,
                                 1, 9,  11, 10, 0,  8,  12, 4,  13, 3, 7,  15, 14, 5,  6,  2,
                                 4, 0,  5,  9,  7,  12, 2,  10, 14, 1, 3,  8,  11, 6,  15, 13};
  static const uint8_t SR[80] = {5,  14, 7,  0, 9, 2,  11, 4,  13, 6,  15, 8,  1,  10, 3,  12,
                                 6,  11, 3,  7, 0, 13, 5,  10, 14, 15, 8,  12, 4,  9,  1,  2,
                                 15, 5,  1,  3, 7, 14, 6,  9,  11, 8,  12, 2,  10, 0,  13, 4,
                                 8,  6,  4,  1, 3, 11, 15, 0,  5,  12, 2,  13, 9,  7,  10, 14,
                                 12, 15, 10, 4, 1, 5,  8,  7,  6,  2,  13, 14, 0,  3,  9,  11};

  for (int blk = 0; blk < 16; ++blk) {
    uint32_t X[16];
    for (int i = 0; i < 16; ++i) X[i] = read_le32(inputs[blk] + i * 4);

    uint32_t al = h0_init, bl = h1_init, cl = h2_init, dl = h3_init, el = h4_init;
    uint32_t ar = h0_init, br = h1_init, cr = h2_init, dr = h3_init, er = h4_init;

    for (int j = 0; j < 80; ++j) {
      uint32_t tl, tr;
      // Left line
      if (j < 16)
        tl = al + (bl ^ cl ^ dl) + X[SL[j]];
      else if (j < 32)
        tl = al + ((bl & cl) | (~bl & dl)) + X[SL[j]] + K[1];
      else if (j < 48)
        tl = al + ((bl | ~cl) ^ dl) + X[SL[j]] + K[2];
      else if (j < 64)
        tl = al + ((bl & dl) | (cl & ~dl)) + X[SL[j]] + K[3];
      else
        tl = al + (bl ^ (cl | ~dl)) + X[SL[j]] + K[4];
      tl = (tl << RL[j] | tl >> (32 - RL[j])) + el;
      al = el;
      el = dl;
      dl = (cl << 10) | (cl >> (32 - 10));
      cl = bl;
      bl = tl;

      // Right line
      if (j < 16)
        tr = ar + (br ^ (cr | ~dr)) + X[SR[j]] + KK[0];
      else if (j < 32)
        tr = ar + ((br & dr) | (cr & ~dr)) + X[SR[j]] + KK[1];
      else if (j < 48)
        tr = ar + ((br | ~cr) ^ dr) + X[SR[j]] + KK[2];
      else if (j < 64)
        tr = ar + ((br & cr) | (~br & dr)) + X[SR[j]] + KK[3];
      else
        tr = ar + (br ^ cr ^ dr) + X[SR[j]];
      tr = (tr << RR[j] | tr >> (32 - RR[j])) + er;
      ar = er;
      er = dr;
      dr = (cr << 10) | (cr >> (32 - 10));
      cr = br;
      br = tr;
    }
    uint32_t t = h1_init + cl + dr;
    uint32_t h1 = h2_init + dl + er;
    uint32_t h2 = h3_init + el + ar;
    uint32_t h3 = h4_init + al + br;
    uint32_t h4 = h0_init + bl + cr;
    uint32_t h0 = t;

    for (int i = 0; i < 4; ++i) {
      outputs[blk][i + 0] = (h0 >> (8 * i)) & 0xFF;
      outputs[blk][i + 4] = (h1 >> (8 * i)) & 0xFF;
      outputs[blk][i + 8] = (h2 >> (8 * i)) & 0xFF;
      outputs[blk][i + 12] = (h3 >> (8 * i)) & 0xFF;
      outputs[blk][i + 16] = (h4 >> (8 * i)) & 0xFF;
    }
  }
}

}  // namespace ripemd160avx512
