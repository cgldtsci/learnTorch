#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMath.c"
#else

#define LAB_IMPLEMENT_BASIC_FUNCTION(NAME, CFUNC)             \
  void THTensor_(NAME)(THTensor *r_, THTensor *t)                \
  {                                                           \
    THTensor_(resizeAs)(r_, t);                               \
    TH_TENSOR_APPLY2(real, t, real, r_, *r__data = CFUNC(*t_data);); \
  }

#if defined(TH_REAL_IS_LONG)
LAB_IMPLEMENT_BASIC_FUNCTION(abs,labs)
#endif /* long only part */

#if defined(TH_REAL_IS_INT)
LAB_IMPLEMENT_BASIC_FUNCTION(abs,abs)
#endif /* int only part */

/* floating point only now */
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
LAB_IMPLEMENT_BASIC_FUNCTION(abs,fabs)

#endif /* floating point only part */

#endif