#include <immintrin.h>
#include <math.h>
#include <string.h>

#include "Int.h"
#include "IntGroup.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

Int _ONE((uint64_t)1);

Int Int::P;

Int::Int() {}

Int::Int(Int *a) {
  if (a)
    Set(a);
  else
    CLEAR();
}

void Int::Xor(const Int *a) {
  if (!a) return;

  int i = 0;
#if defined(__AVX512F__)
  for (; i + 8 <= NB64BLOCK; i += 8) {
    __m512i this_vec = _mm512_load_epi64(bits64 + i);
    __m512i a_vec = _mm512_load_epi64(a->bits64 + i);
    __m512i res = _mm512_xor_epi64(this_vec, a_vec);
    _mm512_store_epi64(bits64 + i, res);
  }
#elif defined(__AVX2__)
  for (; i + 4 <= NB64BLOCK; i += 4) {
    __m256i this_vec = _mm256_load_si256((__m256i *)(bits64 + i));
    __m256i a_vec = _mm256_load_si256((__m256i *)(a->bits64 + i));
    __m256i res = _mm256_xor_si256(this_vec, a_vec);
    _mm256_store_si256((__m256i *)(bits64 + i), res);
  }
#endif
  for (; i < NB64BLOCK; i++) {
    bits64[i] ^= a->bits64[i];
  }
}

Int::Int(int64_t i64) {
  if (i64 < 0) {
    CLEARFF();
  } else {
    CLEAR();
  }
  bits64[0] = i64;
}

Int::Int(uint64_t u64) {
  CLEAR();
  bits64[0] = u64;
}

void Int::CLEAR() {
  int i = 0;
#if defined(__AVX512F__)
  for (; i + 8 <= NB64BLOCK; i += 8) {
    _mm512_store_epi64(bits64 + i, _mm512_setzero_si512());
  }
#endif
#if defined(__AVX2__)
  for (; i + 4 <= NB64BLOCK; i += 4) {
    _mm256_store_si256((__m256i *)(bits64 + i), _mm256_setzero_si256());
  }
#endif
  for (; i < NB64BLOCK; i++) {
    bits64[i] = 0;
  }
}

void Int::CLEARFF() {
  int i = 0;
#if defined(__AVX512F__)
  __m512i ones = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFFULL);
  for (; i + 8 <= NB64BLOCK; i += 8) {
    _mm512_store_epi64(bits64 + i, ones);
  }
#endif
#if defined(__AVX2__)
  __m256i ones256 = _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFFULL);
  for (; i + 4 <= NB64BLOCK; i += 4) {
    _mm256_store_si256((__m256i *)(bits64 + i), ones256);
  }
#endif
  for (; i < NB64BLOCK; i++) {
    bits64[i] = 0xFFFFFFFFFFFFFFFFULL;
  }
}

void Int::Set(Int *a) {
  if (a == this) return;
  int i = 0;
#if defined(__AVX512F__)
  for (; i + 8 <= NB64BLOCK; i += 8) {
    __m512i a_vec = _mm512_load_epi64(a->bits64 + i);
    _mm512_store_epi64(bits64 + i, a_vec);
  }
#endif
#if defined(__AVX2__)
  for (; i + 4 <= NB64BLOCK; i += 4) {
    __m256i a_vec = _mm256_load_si256((__m256i *)(a->bits64 + i));
    _mm256_store_si256((__m256i *)(bits64 + i), a_vec);
  }
#endif
  for (; i < NB64BLOCK; i++) {
    bits64[i] = a->bits64[i];
  }
}

void Int::Add(Int *a) {
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _addcarry_u64(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _addcarry_u64(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _addcarry_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _addcarry_u64(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _addcarry_u64(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _addcarry_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
}

void Int::Add(uint64_t a) {
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a, bits64 + 0);
  c = _addcarry_u64(c, bits64[1], 0, bits64 + 1);
  c = _addcarry_u64(c, bits64[2], 0, bits64 + 2);
  c = _addcarry_u64(c, bits64[3], 0, bits64 + 3);
  c = _addcarry_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], 0, bits64 + 5);
  c = _addcarry_u64(c, bits64[6], 0, bits64 + 6);
  c = _addcarry_u64(c, bits64[7], 0, bits64 + 7);
  c = _addcarry_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

void Int::AddOne() {
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], 1, bits64 + 0);
  c = _addcarry_u64(c, bits64[1], 0, bits64 + 1);
  c = _addcarry_u64(c, bits64[2], 0, bits64 + 2);
  c = _addcarry_u64(c, bits64[3], 0, bits64 + 3);
  c = _addcarry_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], 0, bits64 + 5);
  c = _addcarry_u64(c, bits64[6], 0, bits64 + 6);
  c = _addcarry_u64(c, bits64[7], 0, bits64 + 7);
  c = _addcarry_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

void Int::Add(Int *a, Int *b) {
  unsigned char c = 0;
  c = _addcarry_u64(c, a->bits64[0], b->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, a->bits64[1], b->bits64[1], bits64 + 1);
  c = _addcarry_u64(c, a->bits64[2], b->bits64[2], bits64 + 2);
  c = _addcarry_u64(c, a->bits64[3], b->bits64[3], bits64 + 3);
  c = _addcarry_u64(c, a->bits64[4], b->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, a->bits64[5], b->bits64[5], bits64 + 5);
  c = _addcarry_u64(c, a->bits64[6], b->bits64[6], bits64 + 6);
  c = _addcarry_u64(c, a->bits64[7], b->bits64[7], bits64 + 7);
  c = _addcarry_u64(c, a->bits64[8], b->bits64[8], bits64 + 8);
#endif
}

uint64_t Int::AddCh(Int *a, uint64_t ca, Int *b, uint64_t cb) {
  unsigned char c = 0;
  c = _addcarry_u64(c, a->bits64[0], b->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, a->bits64[1], b->bits64[1], bits64 + 1);
  c = _addcarry_u64(c, a->bits64[2], b->bits64[2], bits64 + 2);
  c = _addcarry_u64(c, a->bits64[3], b->bits64[3], bits64 + 3);
  c = _addcarry_u64(c, a->bits64[4], b->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, a->bits64[5], b->bits64[5], bits64 + 5);
  c = _addcarry_u64(c, a->bits64[6], b->bits64[6], bits64 + 6);
  c = _addcarry_u64(c, a->bits64[7], b->bits64[7], bits64 + 7);
  c = _addcarry_u64(c, a->bits64[8], b->bits64[8], bits64 + 8);
#endif
  uint64_t carry;
  _addcarry_u64(c, ca, cb, &carry);
  return carry;
}

uint64_t Int::AddCh(Int *a, uint64_t ca) {
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _addcarry_u64(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _addcarry_u64(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _addcarry_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _addcarry_u64(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _addcarry_u64(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _addcarry_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
  uint64_t carry;
  _addcarry_u64(c, ca, 0, &carry);
  return carry;
}

uint64_t Int::AddC(Int *a) {
  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _addcarry_u64(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _addcarry_u64(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _addcarry_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _addcarry_u64(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _addcarry_u64(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _addcarry_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
  return c;
}

void Int::AddAndShift(Int *a, Int *b, uint64_t cH) {
  unsigned char c = 0;
  c = _addcarry_u64(c, b->bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, b->bits64[1], a->bits64[1], bits64 + 0);
  c = _addcarry_u64(c, b->bits64[2], a->bits64[2], bits64 + 1);
  c = _addcarry_u64(c, b->bits64[3], a->bits64[3], bits64 + 2);
  c = _addcarry_u64(c, b->bits64[4], a->bits64[4], bits64 + 3);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, b->bits64[5], a->bits64[5], bits64 + 4);
  c = _addcarry_u64(c, b->bits64[6], a->bits64[6], bits64 + 5);
  c = _addcarry_u64(c, b->bits64[7], a->bits64[7], bits64 + 6);
  c = _addcarry_u64(c, b->bits64[8], a->bits64[8], bits64 + 7);
#endif
  bits64[NB64BLOCK - 1] = c + cH;
}

void Int::MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22,
                       uint64_t *cu, uint64_t *cv) {
  Int t1, t2, t3, t4;
  uint64_t c1, c2, c3, c4;
  c1 = t1.IMult(u, _11);
  c2 = t2.IMult(v, _12);
  c3 = t3.IMult(u, _21);
  c4 = t4.IMult(v, _22);
  *cu = u->AddCh(&t1, c1, &t2, c2);
  *cv = v->AddCh(&t3, c3, &t4, c4);
}

void Int::MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22) {
  Int t1, t2, t3, t4;
  t1.IMult(u, _11);
  t2.IMult(v, _12);
  t3.IMult(u, _21);
  t4.IMult(v, _22);
  u->Add(&t1, &t2);
  v->Add(&t3, &t4);
}

bool Int::IsGreater(Int *a) {
  int i = NB64BLOCK - 1;
  while (i >= 0 && a->bits64[i] == bits64[i]) i--;
  return (i >= 0) && (bits64[i] > a->bits64[i]);
}

bool Int::IsLower(Int *a) {
  int i = NB64BLOCK - 1;
  while (i >= 0 && a->bits64[i] == bits64[i]) i--;
  return (i >= 0) && (bits64[i] < a->bits64[i]);
}

bool Int::IsGreaterOrEqual(Int *a) {
  Int p;
  p.Sub(this, a);
  return p.IsPositive();
}

bool Int::IsLowerOrEqual(Int *a) {
  int i = NB64BLOCK - 1;
  while (i >= 0 && a->bits64[i] == bits64[i]) i--;
  if (i >= 0) {
    return bits64[i] < a->bits64[i];
  } else {
    return true;
  }
}

bool Int::IsEqual(Int *a) {
  for (int i = 0; i < NB64BLOCK; i++) {
    if (bits64[i] != a->bits64[i]) return false;
  }
  return true;
}

bool Int::IsOne() { return IsEqual(&_ONE); }

bool Int::IsZero() {
  for (int i = 0; i < NB64BLOCK; i++) {
    if (bits64[i] != 0) return false;
  }
  return true;
}

void Int::SetInt32(uint32_t value) {
  CLEAR();
  bits[0] = value;
}

uint32_t Int::GetInt32() { return bits[0]; }

unsigned char Int::GetByte(int n) { return ((unsigned char *)bits)[n]; }

void Int::Set32Bytes(unsigned char *bytes) {
  CLEAR();
  uint64_t *ptr = (uint64_t *)bytes;
  bits64[3] = _byteswap_uint64(ptr[0]);
  bits64[2] = _byteswap_uint64(ptr[1]);
  bits64[1] = _byteswap_uint64(ptr[2]);
  bits64[0] = _byteswap_uint64(ptr[3]);
}

void Int::Get32Bytes(unsigned char *buff) {
  uint64_t *ptr = (uint64_t *)buff;
  ptr[3] = _byteswap_uint64(bits64[0]);
  ptr[2] = _byteswap_uint64(bits64[1]);
  ptr[1] = _byteswap_uint64(bits64[2]);
  ptr[0] = _byteswap_uint64(bits64[3]);
}

void Int::SetByte(int n, unsigned char byte) { ((unsigned char *)bits)[n] = byte; }

void Int::SetDWord(int n, uint32_t b) { bits[n] = b; }

void Int::SetQWord(int n, uint64_t b) { bits64[n] = b; }

void Int::Sub(Int *a) {
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _subborrow_u64(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _subborrow_u64(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _subborrow_u64(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _subborrow_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _subborrow_u64(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _subborrow_u64(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _subborrow_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
}

void Int::Sub(Int *a, Int *b) {
  unsigned char c = 0;
  c = _subborrow_u64(c, a->bits64[0], b->bits64[0], bits64 + 0);
  c = _subborrow_u64(c, a->bits64[1], b->bits64[1], bits64 + 1);
  c = _subborrow_u64(c, a->bits64[2], b->bits64[2], bits64 + 2);
  c = _subborrow_u64(c, a->bits64[3], b->bits64[3], bits64 + 3);
  c = _subborrow_u64(c, a->bits64[4], b->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, a->bits64[5], b->bits64[5], bits64 + 5);
  c = _subborrow_u64(c, a->bits64[6], b->bits64[6], bits64 + 6);
  c = _subborrow_u64(c, a->bits64[7], b->bits64[7], bits64 + 7);
  c = _subborrow_u64(c, a->bits64[8], b->bits64[8], bits64 + 8);
#endif
}

void Int::Sub(uint64_t a) {
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], a, bits64 + 0);
  c = _subborrow_u64(c, bits64[1], 0, bits64 + 1);
  c = _subborrow_u64(c, bits64[2], 0, bits64 + 2);
  c = _subborrow_u64(c, bits64[3], 0, bits64 + 3);
  c = _subborrow_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], 0, bits64 + 5);
  c = _subborrow_u64(c, bits64[6], 0, bits64 + 6);
  c = _subborrow_u64(c, bits64[7], 0, bits64 + 7);
  c = _subborrow_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

void Int::SubOne() {
  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], 1, bits64 + 0);
  c = _subborrow_u64(c, bits64[1], 0, bits64 + 1);
  c = _subborrow_u64(c, bits64[2], 0, bits64 + 2);
  c = _subborrow_u64(c, bits64[3], 0, bits64 + 3);
  c = _subborrow_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], 0, bits64 + 5);
  c = _subborrow_u64(c, bits64[6], 0, bits64 + 6);
  c = _subborrow_u64(c, bits64[7], 0, bits64 + 7);
  c = _subborrow_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

bool Int::IsPositive() { return (int64_t)(bits64[NB64BLOCK - 1]) >= 0; }

bool Int::IsNegative() { return (int64_t)(bits64[NB64BLOCK - 1]) < 0; }

bool Int::IsStrictPositive() { return IsPositive() && !IsZero(); }

bool Int::IsEven() { return (bits[0] & 0x1) == 0; }

bool Int::IsOdd() { return (bits[0] & 0x1) == 1; }

void Int::Neg() {
  unsigned char c = 0;
  c = _subborrow_u64(c, 0, bits64[0], bits64 + 0);
  c = _subborrow_u64(c, 0, bits64[1], bits64 + 1);
  c = _subborrow_u64(c, 0, bits64[2], bits64 + 2);
  c = _subborrow_u64(c, 0, bits64[3], bits64 + 3);
  c = _subborrow_u64(c, 0, bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, 0, bits64[5], bits64 + 5);
  c = _subborrow_u64(c, 0, bits64[6], bits64 + 6);
  c = _subborrow_u64(c, 0, bits64[7], bits64 + 7);
  c = _subborrow_u64(c, 0, bits64[8], bits64 + 8);
#endif
}

void Int::ShiftL32Bit() {
  for (int i = NB32BLOCK - 1; i > 0; i--) {
    bits[i] = bits[i - 1];
  }
  bits[0] = 0;
}

void Int::ShiftL64Bit() {
  for (int i = NB64BLOCK - 1; i > 0; i--) {
    bits64[i] = bits64[i - 1];
  }
  bits64[0] = 0;
}

void Int::ShiftL64BitAndSub(Int *a, int n) {
  Int b;
  int i = NB64BLOCK - 1;
  for (; i >= n; i--) b.bits64[i] = ~a->bits64[i - n];
  for (; i >= 0; i--) b.bits64[i] = 0xFFFFFFFFFFFFFFFFULL;
  Add(&b);
  AddOne();
}

void Int::ShiftL(uint32_t n) {
  if (n == 0) return;
  if (n < 64) {
    shiftL((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;
    for (uint32_t i = 0; i < nb64; i++) ShiftL64Bit();
    shiftL((unsigned char)nb, bits64);
  }
}

void Int::ShiftR32Bit() {
  for (int i = 0; i < NB32BLOCK - 1; i++) {
    bits[i] = bits[i + 1];
  }
  if (((int32_t)bits[NB32BLOCK - 2]) < 0)
    bits[NB32BLOCK - 1] = 0xFFFFFFFF;
  else
    bits[NB32BLOCK - 1] = 0;
}

void Int::ShiftR64Bit() {
  for (int i = 0; i < NB64BLOCK - 1; i++) {
    bits64[i] = bits64[i + 1];
  }
  if (((int64_t)bits64[NB64BLOCK - 2]) < 0)
    bits64[NB64BLOCK - 1] = 0xFFFFFFFFFFFFFFFF;
  else
    bits64[NB64BLOCK - 1] = 0;
}

void Int::ShiftR(uint32_t n) {
  if (n == 0) return;
  if (n < 64) {
    shiftR((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;
    for (uint32_t i = 0; i < nb64; i++) ShiftR64Bit();
    shiftR((unsigned char)nb, bits64);
  }
}

void Int::SwapBit(int bitNumber) {
  uint32_t nb64 = bitNumber / 64;
  uint32_t nb = bitNumber % 64;
  uint64_t mask = 1ULL << nb;
  if (bits64[nb64] & mask) {
    bits64[nb64] &= ~mask;
  } else {
    bits64[nb64] |= mask;
  }
}

void Int::Mult(Int *a) {
  Int b(this);
  Mult(a, &b);
}

uint64_t Int::IMult(int64_t a) {
  uint64_t carry;
  if (a < 0LL) {
    a = -a;
    Neg();
  }
  imm_imul(bits64, a, bits64, &carry);
  return carry;
}

uint64_t Int::Mult(uint64_t a) {
  uint64_t carry;
  imm_mul(bits64, a, bits64, &carry);
  return carry;
}

uint64_t Int::Mult(Int *a, uint64_t b) {
  uint64_t carry;
  imm_mul(a->bits64, b, bits64, &carry);
  return carry;
}

uint64_t Int::IMult(Int *a, int64_t b) {
  uint64_t carry;
  if (b < 0LL) {
    unsigned char c = 0;
    c = _subborrow_u64(c, 0, a->bits64[0], bits64 + 0);
    c = _subborrow_u64(c, 0, a->bits64[1], bits64 + 1);
    c = _subborrow_u64(c, 0, a->bits64[2], bits64 + 2);
    c = _subborrow_u64(c, 0, a->bits64[3], bits64 + 3);
    c = _subborrow_u64(c, 0, a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
    c = _subborrow_u64(c, 0, a->bits64[5], bits64 + 5);
    c = _subborrow_u64(c, 0, a->bits64[6], bits64 + 6);
    c = _subborrow_u64(c, 0, a->bits64[7], bits64 + 7);
    c = _subborrow_u64(c, 0, a->bits64[8], bits64 + 8);
#endif
    b = -b;
  } else {
    Set(a);
  }
  imm_imul(bits64, b, bits64, &carry);
  return carry;
}

void Int::Mult(Int *a, Int *b) {
  unsigned char c = 0;
  uint64_t h;
  uint64_t pr = 0;
  uint64_t carryh = 0;
  uint64_t carryl = 0;

  bits64[0] = _umul128(a->bits64[0], b->bits64[0], &pr);

  for (int i = 1; i < NB64BLOCK; i++) {
    for (int j = 0; j <= i; j++) {
      c = _addcarry_u64(c, _umul128(a->bits64[j], b->bits64[i - j], &h), pr, &pr);
      c = _addcarry_u64(c, carryl, h, &carryl);
      c = _addcarry_u64(c, carryh, 0, &carryh);
    }
    bits64[i] = pr;
    pr = carryl;
    carryl = carryh;
    carryh = 0;
  }
}

uint64_t Int::Mult(Int *a, uint32_t b) {
#if defined(__BMI2__) && (NB64BLOCK == 5)
  uint64_t a0 = a->bits64[0];
  uint64_t a1 = a->bits64[1];
  uint64_t a2 = a->bits64[2];
  uint64_t a3 = a->bits64[3];
  uint64_t a4 = a->bits64[4];
  uint64_t carry;
  asm volatile(
      "xor %%r10, %%r10              \n\t"
      "mov %[A0], %%rdx              \n\t"
      "mulx %[B], %[R0], %%r9        \n\t"
      "add %%r10, %[R0]              \n\t"
      "adc $0, %%r9                  \n\t"
      "mov %%r9, %%r10               \n\t"
      "mov %[A1], %%rdx              \n\t"
      "mulx %[B], %[R1], %%r9        \n\t"
      "add %%r10, %[R1]              \n\t"
      "adc $0, %%r9                  \n\t"
      "mov %%r9, %%r10               \n\t"
      "mov %[A2], %%rdx              \n\t"
      "mulx %[B], %[R2], %%r9        \n\t"
      "add %%r10, %[R2]              \n\t"
      "adc $0, %%r9                  \n\t"
      "mov %%r9, %%r10               \n\t"
      "mov %[A3], %%rdx              \n\t"
      "mulx %[B], %[R3], %%r9        \n\t"
      "add %%r10, %[R3]              \n\t"
      "adc $0, %%r9                  \n\t"
      "mov %%r9, %%r10               \n\t"
      "mov %[A4], %%rdx              \n\t"
      "mulx %[B], %[R4], %%r9        \n\t"
      "add %%r10, %[R4]              \n\t"
      "adc $0, %%r9                  \n\t"
      "mov %%r9, %[CARRY]            \n\t"
      : [R0] "+r"(bits64[0]), [R1] "+r"(bits64[1]), [R2] "+r"(bits64[2]), [R3] "+r"(bits64[3]),
        [R4] "+r"(bits64[4]), [CARRY] "=r"(carry)
      : [A0] "r"(a0), [A1] "r"(a1), [A2] "r"(a2), [A3] "r"(a3), [A4] "r"(a4), [B] "r"((uint64_t)b)
      : "cc", "rdx", "r9", "r10", "memory");
  return carry;
#else
  __uint128_t c = 0;
  for (int i = 0; i < NB64BLOCK; i++) {
    __uint128_t prod = (__uint128_t(a->bits64[i])) * b + c;
    bits64[i] = (uint64_t)prod;
    c = prod >> 64;
  }
  return (uint64_t)c;
#endif
}

double Int::ToDouble() {
  double base = 1.0;
  double sum = 0;
  double pw32 = pow(2.0, 32.0);
  for (int i = 0; i < NB32BLOCK; i++) {
    sum += (double)(bits[i]) * base;
    base *= pw32;
  }
  return sum;
}

int Int::GetBitLength() {
  Int t(this);
  if (IsNegative()) t.Neg();
  int i = NB64BLOCK - 1;
  while (i >= 0 && t.bits64[i] == 0) i--;
  if (i < 0) return 0;
    return (int)((64 - LZC(t.bits64[i])) + i * 64);
}

int Int::GetSize() {
  int i = NB32BLOCK - 1;
  while (i > 0 && bits[i] == 0) i--;
  return i + 1;
}

int Int::GetSize64() {
  int i = NB64BLOCK - 1;
  while (i > 0 && bits64[i] == 0) i--;
  return i + 1;
}

void Int::MultModN(Int *a, Int *b, Int *n) {
  Int r;
  Mult(a, b);
  Div(n, &r);
  Set(&r);
}

void Int::Mod(Int *n) {
  Int r;
  Div(n, &r);
  Set(&r);
}

int Int::GetLowestBit() {
  int b = 0;
  while (GetBit(b) == 0) b++;
  return b;
}

void Int::MaskByte(int n) {
  for (int i = n; i < NB32BLOCK; i++) bits[i] = 0;
}

void Int::Abs() {
  if (IsNegative()) Neg();
}

void Int::Div(Int *a, Int *mod) {
  if (a->IsGreater(this)) {
    if (mod) mod->Set(this);
    CLEAR();
    return;
  }
  if (a->IsZero()) {
    return;
  }
  if (IsEqual(a)) {
    if (mod) mod->CLEAR();
    Set(&_ONE);
    return;
  }

  Int rem(this);
  Int d(a);
  Int dq;
  CLEAR();

  uint32_t dSize = d.GetSize64();
  uint32_t tSize = rem.GetSize64();
  uint32_t qSize = tSize - dSize + 1;

  uint32_t shift = (uint32_t)LZC(d.bits64[dSize - 1]);
  d.ShiftL(shift);
  rem.ShiftL(shift);

  uint64_t _dh = d.bits64[dSize - 1];
  uint64_t _dl = (dSize > 1) ? d.bits64[dSize - 2] : 0;
  int sb = tSize - 1;

  for (int j = 0; j < (int)qSize; j++) {
    uint64_t nh = rem.bits64[sb - j + 1];
    uint64_t nm = rem.bits64[sb - j];
    uint64_t qhat = 0;
    uint64_t qrem = 0;
    bool skipCorrection = false;

    if (nh == _dh) {
      qhat = ~0ULL;
      qrem = nh + nm;
      skipCorrection = (qrem < nh);
    } else {
      qhat = _udiv128(nh, nm, _dh, &qrem);
    }

    if (qhat) {
      if (!skipCorrection) {
        uint64_t nl = rem.bits64[sb - j - 1];
        uint64_t estProH, estProL;
        estProL = _umul128(_dl, qhat, &estProH);
        if (isStrictGreater128(estProH, estProL, qrem, nl)) {
          qhat--;
          qrem += _dh;
          if (qrem >= _dh) {
            estProL = _umul128(_dl, qhat, &estProH);
            if (isStrictGreater128(estProH, estProL, qrem, nl)) {
              qhat--;
            }
          }
        }
      }
      dq.Mult(&d, qhat);
      rem.ShiftL64BitAndSub(&dq, qSize - j - 1);
      if (rem.IsNegative()) {
        rem.Add(&d);
        qhat--;
      }
    }
    bits64[qSize - j - 1] = qhat;
  }

  if (mod) {
    rem.ShiftR(shift);
    mod->Set(&rem);
  }
}

void Int::GCD(Int *a) {
  uint32_t k;
  uint32_t b;
  Int U(this);
  Int V(a);
  Int T;

  if (U.IsZero()) {
    Set(&V);
    return;
  }
  if (V.IsZero()) {
    Set(&U);
    return;
  }

  if (U.IsNegative()) U.Neg();
  if (V.IsNegative()) V.Neg();

  k = 0;
  while (U.GetBit(k) == 0 && V.GetBit(k) == 0) k++;
  U.ShiftR(k);
  V.ShiftR(k);
  if (U.GetBit(0) == 1) {
    T.Set(&V);
    T.Neg();
  } else {
    T.Set(&U);
  }

  do {
    if (T.IsNegative()) {
      T.Neg();
      b = 0;
      while (T.GetBit(b) == 0) b++;
      T.ShiftR(b);
      V.Set(&T);
      T.Set(&U);
    } else {
      b = 0;
      while (T.GetBit(b) == 0) b++;
      T.ShiftR(b);
      U.Set(&T);
    }
    T.Sub(&V);
  } while (!T.IsZero());

  Set(&U);
  ShiftL(k);
}

void Int::SetBase10(char *value) {
  CLEAR();
  Int pw((uint64_t)1);
  Int c;
  int lgth = (int)strlen(value);
  for (int i = lgth - 1; i >= 0; i--) {
    uint32_t id = (uint32_t)(value[i] - '0');
    c.Set(&pw);
    c.Mult(id);
    Add(&c);
    pw.Mult(10);
  }
}

void Int::SetBase16(char *value) { SetBaseN(16, "0123456789ABCDEF", value); }

std::string Int::GetBase10() { return GetBaseN(10, "0123456789"); }

std::string Int::GetBase16() { return GetBaseN(16, "0123456789ABCDEF"); }

std::string Int::GetBlockStr() {
  char tmp[256];
  char bStr[256];
  tmp[0] = 0;
  for (int i = NB32BLOCK - 3; i >= 0; i--) {
    sprintf(bStr, "%08X", bits[i]);
    strcat(tmp, bStr);
    if (i != 0) strcat(tmp, " ");
  }
  return std::string(tmp);
}

std::string Int::GetC64Str(int nbDigit) {
  char tmp[256];
  char bStr[256];
  tmp[0] = '{';
  tmp[1] = 0;
  for (int i = 0; i < nbDigit; i++) {
    if (bits64[i] != 0) {
      sprintf(bStr, "0x%016" PRIx64 "ULL", bits64[i]);
    } else {
      sprintf(bStr, "0ULL");
    }
    strcat(tmp, bStr);
    if (i != nbDigit - 1) strcat(tmp, ",");
  }
  strcat(tmp, "}");
  return std::string(tmp);
}

void Int::SetBaseN(int n, char *charset, char *value) {
  CLEAR();
  Int pw((uint64_t)1);
  Int nb((uint64_t)n);
  Int c;
  int lgth = (int)strlen(value);
  for (int i = lgth - 1; i >= 0; i--) {
    char *p = strchr(charset, toupper(value[i]));
    if (!p) return;
    int id = (int)(p - charset);
    c.SetInt32(id);
    c.Mult(&pw);
    Add(&c);
    pw.Mult(&nb);
  }
}

std::string Int::GetBaseN(int n, char *charset) {
  std::string ret;
  Int N(this);
  int isNegative = N.IsNegative();
  if (isNegative) N.Neg();

  unsigned char digits[1024] = {0};
  int digitslen = 1;

  for (int i = 0; i < NB64BLOCK * 8; i++) {
    unsigned int carry = N.GetByte(NB64BLOCK * 8 - i - 1);
    for (int j = 0; j < digitslen; j++) {
      carry += (unsigned int)(digits[j]) << 8;
      digits[j] = (unsigned char)(carry % n);
      carry /= n;
    }
    while (carry > 0) {
      digits[digitslen++] = (unsigned char)(carry % n);
      carry /= n;
    }
  }

  if (isNegative) ret.push_back('-');
  for (int i = 0; i < digitslen; i++) ret.push_back(charset[digits[digitslen - 1 - i]]);
  if (ret.length() == 0) ret.push_back('0');

  return ret;
}