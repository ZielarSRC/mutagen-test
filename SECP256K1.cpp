#include <string.h>

#include "SECP256K1.h"

Secp256K1::Secp256K1() {}

void Secp256K1::Init() {
  // Prime for the finite field
  Int P;
  P.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

  // Set up field
  Int::SetupField(&P);

  // Generator point and order
  G.x.SetBase16("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  G.y.SetBase16("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
  G.z.SetInt32(1);
  order.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

  Int::InitK1(&order);

  // Compute Generator table
  Point N(G);
  for (int i = 0; i < 32; i++) {
    GTable[i * 256].x = N.x;
    GTable[i * 256].y = N.y;
    N = DoubleDirect(N);
    for (int j = 1; j < 255; j++) {
      GTable[i * 256 + j].x = N.x;
      GTable[i * 256 + j].y = N.y;
      // Konwersja AffinePoint na Point
      Point basePoint;
      basePoint.x = GTable[i * 256].x;
      basePoint.y = GTable[i * 256].y;
      basePoint.z.SetInt32(1);
      N = AddDirect(N, basePoint);
    }
    GTable[i * 256 + 255].x = N.x;  // Dummy point
    GTable[i * 256 + 255].y = N.y;
  }
}

Secp256K1::~Secp256K1() {}

Point Secp256K1::AddDirect(Point &p1, Point &p2) {
  Int _s;
  Int _p;
  Int dy;
  Int dx;
  Point r;
  r.z.SetInt32(1);

  dy.ModSub(&p2.y, &p1.y);
  dx.ModSub(&p2.x, &p1.x);
  dx.ModInv();
  _s.ModMulK1(&dy, &dx);  // s = (p2.y-p1.y)*inverse(p2.x-p1.x);

  _p.ModSquareK1(&_s);  // _p = pow2(s)

  r.x.ModSub(&_p, &p1.x);
  r.x.ModSub(&p2.x);  // rx = pow2(s) - p1.x - p2.x;

  r.y.ModSub(&p2.x, &r.x);
  r.y.ModMulK1(&_s);
  r.y.ModSub(&p2.y);  // ry = - p2.y - s*(ret.x-p2.x);

  return r;
}

Point Secp256K1::Add2(Point &p1, Point &p2) {
  // P2.z = 1

  Int u;
  Int v;
  Int u1;
  Int v1;
  Int vs2;
  Int vs3;
  Int us2;
  Int a;
  Int us2w;
  Int vs2v2;
  Int vs3u2;
  Int _2vs2v2;
  Point r;

  u1.ModMulK1(&p2.y, &p1.z);
  v1.ModMulK1(&p2.x, &p1.z);
  u.ModSub(&u1, &p1.y);
  v.ModSub(&v1, &p1.x);
  us2.ModSquareK1(&u);
  vs2.ModSquareK1(&v);
  vs3.ModMulK1(&vs2, &v);
  us2w.ModMulK1(&us2, &p1.z);
  vs2v2.ModMulK1(&vs2, &p1.x);
  _2vs2v2.ModAdd(&vs2v2, &vs2v2);
  a.ModSub(&us2w, &vs3);
  a.ModSub(&_2vs2v2);

  r.x.ModMulK1(&v, &a);

  vs3u2.ModMulK1(&vs3, &p1.y);
  r.y.ModSub(&vs2v2, &a);
  r.y.ModMulK1(&r.y, &u);
  r.y.ModSub(&vs3u2);

  r.z.ModMulK1(&vs3, &p1.z);

  return r;
}

Point Secp256K1::AddAffine(Point &p1, Int &x2, Int &y2) {
  Int u;
  Int v;
  Int u1;
  Int v1;
  Int vs2;
  Int vs3;
  Int us2;
  Int a;
  Int us2w;
  Int vs2v2;
  Int vs3u2;
  Int _2vs2v2;
  Point r;

  u1.ModMulK1(&y2, &p1.z);
  v1.ModMulK1(&x2, &p1.z);
  u.ModSub(&u1, &p1.y);
  v.ModSub(&v1, &p1.x);
  us2.ModSquareK1(&u);
  vs2.ModSquareK1(&v);
  vs3.ModMulK1(&vs2, &v);
  us2w.ModMulK1(&us2, &p1.z);
  vs2v2.ModMulK1(&vs2, &p1.x);
  _2vs2v2.ModAdd(&vs2v2, &vs2v2);
  a.ModSub(&us2w, &vs3);
  a.ModSub(&_2vs2v2);

  r.x.ModMulK1(&v, &a);

  vs3u2.ModMulK1(&vs3, &p1.y);
  r.y.ModSub(&vs2v2, &a);
  r.y.ModMulK1(&r.y, &u);
  r.y.ModSub(&vs3u2);

  r.z.ModMulK1(&vs3, &p1.z);

  return r;
}

Point Secp256K1::Add(Point &p1, Point &p2) {
  Int u, v;
  Int u1, u2;
  Int v1, v2;
  Int vs2, vs3;
  Int us2, w;
  Int a, us2w;
  Int vs2v2, vs3u2;
  Int _2vs2v2, x3;
  Int vs3y1;
  Point r;

  // Вычисляем промежуточные значения
  u1.ModMulK1(&p2.y, &p1.z);
  u2.ModMulK1(&p1.y, &p2.z);
  v1.ModMulK1(&p2.x, &p1.z);
  v2.ModMulK1(&p1.x, &p2.z);

  // Проверка на точку на бесконечности
  if (v1.IsEqual(&v2)) {     // Проверяем, равны ли X-координаты
    if (!u1.IsEqual(&u2)) {  // Если Y-координаты разные
      // Точка на бесконечности
      r.x.SetInt32(0);
      r.y.SetInt32(0);
      r.z.SetInt32(0);
      return r;
    } else {
      // Удвоение точки
      return Double(p1);  // Метод для удвоения точки
    }
  }

  // Основные вычисления сложения точек
  u.ModSub(&u1, &u2);
  v.ModSub(&v1, &v2);
  w.ModMulK1(&p1.z, &p2.z);
  us2.ModSquareK1(&u);
  vs2.ModSquareK1(&v);
  vs3.ModMulK1(&vs2, &v);
  us2w.ModMulK1(&us2, &w);
  vs2v2.ModMulK1(&vs2, &v2);
  _2vs2v2.ModAdd(&vs2v2, &vs2v2);
  a.ModSub(&us2w, &vs3);
  a.ModSub(&_2vs2v2);
  r.x.ModMulK1(&v, &a);
  vs3u2.ModMulK1(&vs3, &u2);
  r.y.ModSub(&vs2v2, &a);
  r.y.ModMulK1(&r.y, &u);
  r.y.ModSub(&vs3u2);
  r.z.ModMulK1(&vs3, &w);
  return r;
}

Point Secp256K1::DoubleDirect(Point &p) {
  Int _s;
  Int _p;
  Int a;
  Point r;
  r.z.SetInt32(1);

  _s.ModSquareK1(&p.x);
  _p.ModAdd(&_s, &_s);
  _p.ModAdd(&_s);

  a.ModAdd(&p.y, &p.y);
  a.ModInv();
  _s.ModMulK1(&_p, &a);  // s = (3*pow2(p.x))*inverse(2*p.y);

  _p.ModMulK1(&_s, &_s);
  a.ModAdd(&p.x, &p.x);
  a.ModNeg();
  r.x.ModAdd(&a, &_p);  // rx = pow2(s) + neg(2*p.x);

  a.ModSub(&r.x, &p.x);

  _p.ModMulK1(&a, &_s);
  r.y.ModAdd(&_p, &p.y);
  r.y.ModNeg();  // ry = neg(p.y + s*(ret.x+neg(p.x)));

  return r;
}

Point Secp256K1::ComputePublicKey(Int *privKey) {
  int i = 0;
  uint8_t b;
  Point Q;
  Q.Clear();

  // Search first significant byte
  for (i = 0; i < 32; i++) {
    b = privKey->GetByte(i);
    if (b) break;
  }
  Q.x = GTable[256 * i + (b - 1)].x;
  Q.y = GTable[256 * i + (b - 1)].y;
  Q.z.SetInt32(1);
  i++;

  for (; i < 32; i++) {
    b = privKey->GetByte(i);
    if (b) Q = AddAffine(Q, GTable[256 * i + (b - 1)].x, GTable[256 * i + (b - 1)].y);
  }

  Q.Reduce();
  return Q;
}

Point Secp256K1::Double(Point &p) {
  Int x2;
  Int _3x2;
  Int w;
  Int s;
  Int s2;
  Int b;
  Int _8b;
  Int _8y2s2;
  Int y2;
  Int h;
  Point r;

  // w = 3 * x^2 (since a=0 for secp256k1)
  x2.ModSquareK1(&p.x);
  _3x2.ModAdd(&x2, &x2);  // 2*x^2
  _3x2.ModAdd(&x2);       // 3*x^2
  w = _3x2;

  s.ModMulK1(&p.y, &p.z);
  b.ModMulK1(&p.y, &s);  // b = y^2 * z
  b.ModMulK1(&p.x);      // b = x * y^2 * z

  h.ModSquareK1(&w);
  _8b.ModAdd(&b, &b);      // 2b
  _8b.ModAdd(&_8b, &_8b);  // 4b
  _8b.ModAdd(&_8b, &_8b);  // 8b
  h.ModSub(&_8b);

  r.x.ModMulK1(&h, &s);
  r.x.ModAdd(&r.x, &r.x);  // 2*h*s

  s2.ModSquareK1(&s);
  y2.ModSquareK1(&p.y);
  _8y2s2.ModMulK1(&y2, &s2);
  _8y2s2.ModAdd(&_8y2s2, &_8y2s2);  // 2
  _8y2s2.ModAdd(&_8y2s2, &_8y2s2);  // 4
  _8y2s2.ModAdd(&_8y2s2, &_8y2s2);  // 8

  r.y.ModAdd(&b, &b);      // 2b
  r.y.ModAdd(&r.y, &r.y);  // 4b
  r.y.ModSub(&h);          // 4b - h
  r.y.ModMulK1(&w);
  r.y.ModSub(&_8y2s2);

  r.z.ModMulK1(&s2, &s);   // s^3
  r.z.ModAdd(&r.z, &r.z);  // 2*s^3
  r.z.ModAdd(&r.z, &r.z);  // 4*s^3
  r.z.ModAdd(&r.z, &r.z);  // 8*s^3

  return r;
}

Int Secp256K1::GetY(Int x, bool isEven) {
  Int _s;
  Int _p;

  _s.ModSquareK1(&x);
  _p.ModMulK1(&_s, &x);
  _p.ModAdd(7);
  _p.ModSqrt();

  if (!_p.IsEven() && isEven) {
    _p.ModNeg();
  } else if (_p.IsEven() && !isEven) {
    _p.ModNeg();
  }

  return _p;
}

bool Secp256K1::EC(Point &p) {
  Int _s;
  Int _p;

  _s.ModSquareK1(&p.x);
  _p.ModMulK1(&_s, &p.x);
  _p.ModAdd(7);
  _s.ModMulK1(&p.y, &p.y);
  _s.ModSub(&_p);

  return _s.IsZero();  // ( ((pow2(y) - (pow3(x) + 7)) % P) == 0 );
}