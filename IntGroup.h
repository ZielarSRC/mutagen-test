#include <immintrin.h>

#include <cstdlib>

#include "IntGroup.h"

IntGroup::IntGroup(int size) {
  this->size = size;
  subp = (Int *)_mm_malloc(size * sizeof(Int), 64);
}

IntGroup::~IntGroup() { _mm_free(subp); }

void IntGroup::Set(Int *pts) { ints = pts; }

void IntGroup::ModInv() {
  Int newValue;
  Int inverse;

  subp[0].Set(&ints[0]);
  for (int i = 1; i < size; i++) {
    subp[i].ModMulK1(&subp[i - 1], &ints[i]);
  }

  inverse.Set(&subp[size - 1]);
  inverse.ModInv();

  for (int i = size - 1; i > 0; i--) {
    newValue.ModMulK1(&subp[i - 1], &inverse);
    inverse.ModMulK1(&inverse, &ints[i]);
    ints[i].Set(&newValue);
  }

  ints[0].Set(&inverse);
}
