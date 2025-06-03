#include "Point.h"

Point::Point() {
  // Inicjalizacja z domyślnymi wartościami
  x.SetInt32(0);
  y.SetInt32(0);
  z.SetInt32(1);
}

Point::Point(const Point &p) {
  // Kopiowanie z wykorzystaniem AVX
  const uint64_t *src = (const uint64_t *)&p;
  uint64_t *dst = (uint64_t *)this;

  for (int i = 0; i < sizeof(Point) / sizeof(uint64_t); i += 8) {
    __m512i data = _mm512_load_epi64(src + i);
    _mm512_store_epi64(dst + i, data);
  }
}

Point::Point(Int *cx, Int *cy, Int *cz) {
  // Ustawienie wartości z optymalizacją pamięciową
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::Point(Int *cx, Int *cz) {
  x.Set(cx);
  z.Set(cz);
  y.SetInt32(0);
}

void Point::Clear() {
  x.SetInt32(0);
  y.SetInt32(0);
  z.SetInt32(0);
}

void Point::Set(Int *cx, Int *cy, Int *cz) {
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::~Point() {
  // Brak specjalnych operacji
}

void Point::Set(Point &p) {
  // Kopiowanie z wykorzystaniem wektorów
  __m512i px = _mm512_load_epi64((__m512i *)&p.x);
  __m512i py = _mm512_load_epi64((__m512i *)&p.y);
  __m512i pz = _mm512_load_epi64((__m512i *)&p.z);

  _mm512_store_epi64((__m512i *)&x, px);
  _mm512_store_epi64((__m512i *)&y, py);
  _mm512_store_epi64((__m512i *)&z, pz);
}

bool Point::isZero() { return x.IsZero() && y.IsZero(); }

void Point::Reduce() {
  if (z.IsOne()) return;  // Optymalizacja częstego przypadku

  Int zi;
  zi.Set(&z);   // Skopiuj z do zi
  zi.ModInv();  // zi = 1/z

  // Obliczenia równoległe
  x.ModMul(&x, &zi);
  y.ModMul(&y, &zi);
  z.SetInt32(1);
}

bool Point::equals(Point &p) {
  // Porównanie z wykorzystaniem XOR
  __mmask8 eqX =
      _mm512_cmpeq_epi64_mask(_mm512_load_epi64((__m512i *)&x), _mm512_load_epi64((__m512i *)&p.x));

  __mmask8 eqY =
      _mm512_cmpeq_epi64_mask(_mm512_load_epi64((__m512i *)&y), _mm512_load_epi64((__m512i *)&p.y));

  __mmask8 eqZ =
      _mm512_cmpeq_epi64_mask(_mm512_load_epi64((__m512i *)&z), _mm512_load_epi64((__m512i *)&p.z));

  return (eqX == 0xFF) && (eqY == 0xFF) && (eqZ == 0xFF);
}