#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensor.h"
#else

/* a la lua? dim, storageoffset, ...  et les methodes ? */

#define TH_TENSOR_REFCOUNTED 1

typedef struct THTensor
{
    long *size;
    long *stride;
    int nDimension;

    THStorage *storage;
    long storageOffset;
    int refcount;

    char flag;

} THTensor;

/**** creation methods ****/
TH_API THTensor *THTensor_(new)(void);
TH_API THTensor *THTensor_(newWithTensor)(THTensor *tensor);

/* stride might be NULL */
TH_API THTensor *THTensor_(newWithStorage)(THStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);

/* stride might be NULL */
TH_API THTensor *THTensor_(newWithSize)(THLongStorage *size_, THLongStorage *stride_);
TH_API void THTensor_(free)(THTensor *self);

#endif
