#ifndef INTGROUPH
#define INTGROUPH

#include <cstdlib>  // Dodano dla size_t

#include "Int.h"

class IntGroup {
 public:
  IntGroup(int size);
  ~IntGroup();
  void Set(Int *pts);
  void ModInv();

 private:
  Int *ints;
  Int *subp;
  int size;
};

#endif  // INTGROUPH