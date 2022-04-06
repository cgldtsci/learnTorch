#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMath.h"
#else

TH_API void THTensor_(fill)(THTensor *r_, real value);
TH_API void THTensor_(neg)(THTensor *self, THTensor *src);

TH_API void THTensor_(add)(THTensor *r_, THTensor *t, real value);
TH_API void THTensor_(sub)(THTensor *self, THTensor *src, real value);
TH_API void THTensor_(mul)(THTensor *r_, THTensor *t, real value);
TH_API void THTensor_(div)(THTensor *r_, THTensor *t, real value);
TH_API void THTensor_(remainder)(THTensor *r_, THTensor *t, real value);

TH_API void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src);
TH_API void THTensor_(csub)(THTensor *self, THTensor *src1, real value, THTensor *src2);
TH_API void THTensor_(cmul)(THTensor *r_, THTensor *t, THTensor *src);
TH_API void THTensor_(cdiv)(THTensor *r_, THTensor *t, THTensor *src);
TH_API void THTensor_(cremainder)(THTensor *r_, THTensor *t, THTensor *src);

TH_API real THTensor_(minall)(THTensor *t);
TH_API real THTensor_(maxall)(THTensor *t);

TH_API long THTensor_(numel)(THTensor *t);

TH_API void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension);
TH_API void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension);

TH_API void THTensor_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat,  THTensor *vec);
TH_API void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat1, THTensor *mat2);

#if defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
TH_API void THTensor_(abs)(THTensor *r_, THTensor *t);
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

TH_API void THTensor_(abs)(THTensor *r_, THTensor *t);

#endif



#endif