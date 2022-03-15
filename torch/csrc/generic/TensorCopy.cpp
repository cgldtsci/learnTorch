#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TensorCopy.cpp"
#else

#define IMPLEMENT_COPY_WRAPPER(NAME,TYPEA,TYPEB)                               \
    IMPLEMENT_COPY_WRAPPER_FULLNAME(THTensor_(NAME), TYPEA, TYPEB)

#define IMPLEMENT_COPY_WRAPPER_FULLNAME(NAME,TYPEA,TYPEB)                      \
void TH_CONCAT_2(_THPCopy_,NAME)(PyObject *dst, PyObject *src)                 \
{                                                                              \
  NAME(LIBRARY_STATE ((TYPEA *)dst)->cdata,                                    \
          ((TYPEB *)src)->cdata);                                              \
}

IMPLEMENT_COPY_WRAPPER(copy,        THPTensor,          THPTensor)
IMPLEMENT_COPY_WRAPPER(copyByte,    THPTensor,          THPByteTensor)
IMPLEMENT_COPY_WRAPPER(copyChar,    THPTensor,          THPCharTensor)
IMPLEMENT_COPY_WRAPPER(copyShort,   THPTensor,          THPShortTensor)
IMPLEMENT_COPY_WRAPPER(copyInt,     THPTensor,          THPIntTensor)
IMPLEMENT_COPY_WRAPPER(copyLong,    THPTensor,          THPLongTensor)
IMPLEMENT_COPY_WRAPPER(copyFloat,   THPTensor,          THPFloatTensor)
IMPLEMENT_COPY_WRAPPER(copyDouble,  THPTensor,          THPDoubleTensor)

#endif
