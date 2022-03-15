#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/StorageCopy.cpp"
#else

#define IMPLEMENT_COPY_WRAPPER(NAME,TYPEA,TYPEB)                               \
    IMPLEMENT_COPY_WRAPPER_FULLNAME(THStorage_(NAME), TYPEA, TYPEB)

#define IMPLEMENT_COPY_WRAPPER_FULLNAME(NAME,TYPEA,TYPEB)                      \
void TH_CONCAT_2(_THPCopy_,NAME)(PyObject *dst, PyObject *src)                 \
{                                                                              \
  NAME(LIBRARY_STATE ((TYPEA *)dst)->cdata,                                    \
          ((TYPEB *)src)->cdata);                                              \
}

IMPLEMENT_COPY_WRAPPER(copy,        THPStorage,          THPStorage)
IMPLEMENT_COPY_WRAPPER(copyByte,    THPStorage,          THPByteStorage)
IMPLEMENT_COPY_WRAPPER(copyChar,    THPStorage,          THPCharStorage)
IMPLEMENT_COPY_WRAPPER(copyShort,   THPStorage,          THPShortStorage)
IMPLEMENT_COPY_WRAPPER(copyInt,     THPStorage,          THPIntStorage)
IMPLEMENT_COPY_WRAPPER(copyLong,    THPStorage,          THPLongStorage)
IMPLEMENT_COPY_WRAPPER(copyFloat,   THPStorage,          THPFloatStorage)
IMPLEMENT_COPY_WRAPPER(copyDouble,  THPStorage,          THPDoubleStorage)

#endif
