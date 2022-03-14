#ifndef THP_WRAP_UTILS_INC
#define THP_WRAP_UTILS_INC

#define THPUtils_(NAME) TH_CONCAT_4(THP,Real,Utils_,NAME)

void THPUtils_setError(const char *format, ...);

#define THStoragePtr TH_CONCAT_3(TH,Real,StoragePtr)
#define THTensorPtr  TH_CONCAT_3(TH,Real,TensorPtr)
#define THPStoragePtr TH_CONCAT_3(THP,Real,StoragePtr)
#define THPTensorPtr  TH_CONCAT_3(THP,Real,TensorPtr)

bool THPUtils_checkLong(PyObject *index);
int THPUtils_getLong(PyObject *index, long *result);
long THPUtils_unpackLong(PyObject *index);
THLongStorage * THPUtils_getLongStorage(PyObject *args, int ignore_first=0);

template<class T>
class THPPointer {

public:
  THPPointer(): ptr(nullptr) {};
  THPPointer(T *ptr): ptr(ptr) {};
//  THPPointer(THPPointer &&p) { free(); ptr = p.ptr; p.ptr = nullptr; };

  ~THPPointer() { free(); };
  T * get() { return ptr; }
  T * release() { T *tmp = ptr; ptr = NULL; return tmp; }
  // implit transform
  operator T*() { return ptr; }
//https://stackoverflow.com/questions/2750316/this-vs-this-in-c
  THPPointer& operator =(T *new_ptr) { free(); ptr = new_ptr; return *this; }
  T * operator ->() { return ptr; }

private:
  void free();
  T *ptr = nullptr;
};

#include "generic/utils.h"
#include <TH/THGenerateAllTypes.h>

typedef THPPointer<PyObject> THPObjectPtr;

#endif
