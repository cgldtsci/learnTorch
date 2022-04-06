#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMath.c"
#else

void THTensor_(fill)(THTensor *r_, real value)
{
  TH_TENSOR_APPLY(real, r_,
                  THVector_(fill)(r__data, value, r__size); break;);
}

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

#undef th_isnan
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
#define th_isnan(val) \
if (isnan(value)) break;
#else
#define th_isnan(val)
#endif

real THTensor_(minall)(THTensor *tensor)
{
  real theMin;
  real value;

  THArgCheck(tensor->nDimension > 0, 1, "tensor must have one dimension");
  theMin = THTensor_(data)(tensor)[0];
  TH_TENSOR_APPLY(real, tensor,
                  value = *tensor_data;
                  /* This is not the same as value<theMin in the case of NaNs */
                  if(!(value >= theMin))
                  {
                    theMin = value;
                    th_isnan(value)
                  });
  return theMin;
}


real THTensor_(maxall)(THTensor *tensor)
{
  real theMax;
  real value;

  THArgCheck(tensor->nDimension > 0, 1, "tensor must have one dimension");
  theMax = THTensor_(data)(tensor)[0];
  TH_TENSOR_APPLY(real, tensor,
                  value = *tensor_data;
                  /* This is not the same as value>theMax in the case of NaNs */
                  if(!(value <= theMax))
                  {
                    theMax = value;
                    th_isnan(value)
                  });
  return theMax;
}

void THTensor_(min)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension)
{
  THLongStorage *dim;
  real theMin;
  real value;
  long theIndex;
  long i;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(values_, dim, NULL);
  THLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY3(real, t, real, values_, long, indices_, dimension,
                       theMin = t_data[0];
                       theIndex = 0;

                       for(i = 0; i < t_size; i++)
                       {
                         value = t_data[i*t_stride];
                         /* This is not the same as value<theMin in the case of NaNs */
                         if(!(value >= theMin))
                         {
                           theIndex = i;
                           theMin = value;
                           th_isnan(value)
                         }
                       }
                       *indices__data = theIndex;
                       *values__data = theMin;);
}


void THTensor_(max)(THTensor *values_, THLongTensor *indices_, THTensor *t, int dimension)
{
  THLongStorage *dim;
  real theMax;
  real value;
  long theIndex;
  long i;

  THArgCheck(dimension >= 0 && dimension < THTensor_(nDimension)(t), 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  dim = THTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THTensor_(resize)(values_, dim, NULL);
  THLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY3(real, t, real, values_, long, indices_, dimension,
                       theMax = t_data[0];
                       theIndex = 0;

                       for(i = 0; i < t_size; i++)
                       {
                         value = t_data[i*t_stride];
                         /* This is not the same as value>theMax in the case of NaNs */
                         if(!(value <= theMax))
                         {
                           theIndex = i;
                           theMax = value;
                           th_isnan(value)
                         }
                       }
                       *indices__data = theIndex;
                       *values__data = theMax;);
}

long THTensor_(numel)(THTensor *t)
{
  return THTensor_(nElement)(t);
}


void THTensor_(add)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++)
          rp[i] = tp[i] + value;
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data + value;);
  }
}

void THTensor_(sub)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(add)(r_, t, -value);
}

void THTensor_(cadd)(THTensor *r_, THTensor *t, real value, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(isContiguous)(src) && THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
    if(r_ == t) {
      THBlas_(axpy)(THTensor_(nElement)(t), value, THTensor_(data)(src), 1, THTensor_(data)(r_), 1);
    } else {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i< sz; i++)
          rp[i] = tp[i] + value * sp[i];
    }
  } else {
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data + value * *src_data;);
  }
}

void THTensor_(csub)(THTensor *r_, THTensor *t, real value,THTensor *src)
{
  THTensor_(cadd)(r_, t, -value, src);
}

#endif