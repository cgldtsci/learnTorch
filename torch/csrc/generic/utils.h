#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/utils.h"
#else

struct THPStorage;

typedef class THPPointer<THStorage>      THStoragePtr;
typedef class THPPointer<THPStorage>      THPStoragePtr;

bool THPUtils_(parseSlice)(PyObject *slice, Py_ssize_t len, Py_ssize_t *ostart, Py_ssize_t *ostop, Py_ssize_t *oslicelength);
bool THPUtils_(parseReal)(PyObject *value, real *result);
real THPUtils_(unpackReal)(PyObject *value);

#endif
