#ifndef THP_WRAP_UTILS_INC
#define THP_WRAP_UTILS_INC

void THPUtils_setError(const char *format, ...);

#define THPStoragePtr TH_CONCAT_3(THP,Real,StoragePtr)

template<class T>
class THPPointer {

public:
  THPPointer(): ptr(nullptr) {};
  THPPointer(T *ptr): ptr(ptr) {};
//  THPPointer(THPPointer &&p) { free(); ptr = p.ptr; p.ptr = nullptr; };
//  T * release() { T *tmp = ptr; ptr = NULL; return tmp; }
//https://stackoverflow.com/questions/2750316/this-vs-this-in-c
  THPPointer& operator =(T *new_ptr) { free(); ptr = new_ptr; return *this; }

private:
  void free();
  T *ptr = nullptr;
};

#include "generic/utils.h"
#include <TH/THGenerateAllTypes.h>

#endif
