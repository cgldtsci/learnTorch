#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMath.h"
#else

#if defined(TH_REAL_IS_INT) || defined(TH_REAL_IS_LONG)
TH_API void THTensor_(abs)(THTensor *r_, THTensor *t);
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

TH_API void THTensor_(abs)(THTensor *r_, THTensor *t);

#endif

#endif