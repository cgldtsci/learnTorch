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

/**** access methods ****/
TH_API real *THTensor_(data)(const THTensor *self);
TH_API int THTensor_(nDimension)(const THTensor *self);
TH_API THStorage* THTensor_(storage)(const THTensor *self);
TH_API THLongStorage *THTensor_(newSizeOf)(THTensor *self);
TH_API long THTensor_(size)(const THTensor *self, int dim);
TH_API long THTensor_(storageOffset)(const THTensor *self);
TH_API long THTensor_(stride)(const THTensor *self, int dim);
TH_API THLongStorage *THTensor_(newStrideOf)(THTensor *self);


/**** creation methods ****/
TH_API THTensor *THTensor_(new)(void);
TH_API THTensor *THTensor_(newWithTensor)(THTensor *tensor);

/* stride might be NULL */
TH_API THTensor *THTensor_(newWithStorage)(THStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);

/* stride might be NULL */

TH_API THTensor *THTensor_(newWithSize)(THLongStorage *size_, THLongStorage *stride_);
TH_API void THTensor_(resize)(THTensor *tensor, THLongStorage *size, THLongStorage *stride);
TH_API void THTensor_(resizeAs)(THTensor *tensor, THTensor *src);
TH_API void THTensor_(free)(THTensor *self);

TH_API THTensor *THTensor_(newClone)(THTensor *self);
TH_API THTensor *THTensor_(newTranspose)(THTensor *tensor, int dimension1_, int dimension2_);
TH_API THTensor *THTensor_(newContiguous)(THTensor *tensor);

TH_API void THTensor_(set)(THTensor *self, THTensor *src);

TH_API real THTensor_(get1d)(const THTensor *tensor, long x0);

TH_API int THTensor_(isContiguous)(const THTensor *self);

TH_API void THTensor_(resize1d)(THTensor *tensor, long size0_);
TH_API void THTensor_(resize4d)(THTensor *tensor, long size0_, long size1_, long size2_, long size3_);

TH_API long THTensor_(nElement)(const THTensor *self);

TH_API int THTensor_(isSameSizeAs)(const THTensor *self, const THTensor *src);

TH_API void THTensor_(setStorage)(THTensor *self, THStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);

TH_API void THTensor_(narrow)(THTensor *self, THTensor *src, int dimension_, long firstIndex_, long size_);
TH_API void THTensor_(select)(THTensor *self, THTensor *src, int dimension_, long sliceIndex_);
TH_API void THTensor_(transpose)(THTensor *self, THTensor *src, int dimension1_, int dimension2_);
TH_API void THTensor_(retain)(THTensor *self);
TH_API void THTensor_(unfold)(THTensor *self, THTensor *src, int dimension_, long size_, long step_);

#endif
