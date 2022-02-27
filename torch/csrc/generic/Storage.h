#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Storage.h"
#else

struct THPStorage {
  PyObject_HEAD
//  THStorage *cdata;
};

bool THPStorage_(init)(PyObject *module);

#endif
