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
LAB_IMPLEMENT_BASIC_FUNCTION(neg,-)


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


void THTensor_(mul)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++)
          rp[i] = tp[i] * value;
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data * value;);
  }
}

void THTensor_(div)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++)
          rp[i] = tp[i] / value;
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data / value;);
  }
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


void THTensor_(addmv)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *mat, THTensor *vec)
{
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected, got %dD, %dD",
      mat->nDimension, vec->nDimension);

  if( mat->size[1] != vec->size[0] ) {
    THDescBuff bm = THTensor_(sizeDesc)(mat);
    THDescBuff bv = THTensor_(sizeDesc)(vec);
    THError("size mismatch, %s, %s", bm.str, bv.str);
  }

  if(t->nDimension != 1)
    THError("vector expected, got t: %dD", t->nDimension);

  if(t->size[0] != mat->size[0]) {
    THDescBuff bt = THTensor_(sizeDesc)(t);
    THDescBuff bm = THTensor_(sizeDesc)(mat);
    THError("size mismatch, t: %s, mat: %s", bt.str, bm.str);
  }

  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }

  if(mat->stride[0] == 1)
  {
    THBlas_(gemv)('n', mat->size[0], mat->size[1],
                  alpha, THTensor_(data)(mat), mat->stride[1],
                  THTensor_(data)(vec), vec->stride[0],
                  beta, THTensor_(data)(r_), r_->stride[0]);
  }
  else if(mat->stride[1] == 1)
  {
    THBlas_(gemv)('t',  mat->size[1], mat->size[0],
                  alpha, THTensor_(data)(mat), mat->stride[0],
                  THTensor_(data)(vec), vec->stride[0],
                  beta, THTensor_(data)(r_), r_->stride[0]);
  }
  else
  {
    THTensor *cmat = THTensor_(newContiguous)(mat);

    THBlas_(gemv)('t',  mat->size[1], mat->size[0],
                  alpha, THTensor_(data)(cmat), cmat->stride[0],
                  THTensor_(data)(vec), vec->stride[0],
                  beta, THTensor_(data)(r_), r_->stride[0]);

    THTensor_(free)(cmat);
  }
}

void THTensor_(addmm)(THTensor *r_, real beta, THTensor *t, real alpha, THTensor *m1, THTensor *m2)
{
  char transpose_r, transpose_m1, transpose_m2;
  THTensor *r__, *m1_, *m2_;

  if( (m1->nDimension != 2) || (m2->nDimension != 2))
    THError("matrices expected, got %dD, %dD tensors", m1->nDimension, m2->nDimension);

  if(m1->size[1] != m2->size[0]) {
    THDescBuff bm1 = THTensor_(sizeDesc)(m1);
    THDescBuff bm2 = THTensor_(sizeDesc)(m2);
    THError("size mismatch, m1: %s, m2: %s", bm1.str, bm2.str);
  }

  if( t->nDimension != 2 )
    THError("matrix expected, got %dD tensor for t", t->nDimension);

  if( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) ) {
    THDescBuff bt  = THTensor_(sizeDesc)(t);
    THDescBuff bm1 = THTensor_(sizeDesc)(m1);
    THDescBuff bm2 = THTensor_(sizeDesc)(m2);
    THError("size mismatch, t: %s, m1: %s, m2: %s", bt.str, bm1.str, bm2.str);
  }

  if(t != r_)
  {
    THTensor_(resizeAs)(r_, t);
    THTensor_(copy)(r_, t);
  }

/*  printf("%ldx%ld = %ldx%ld X %ldx%ld\n", r_->size[0], r_->size[1], m1->size[0], m1->size[1], m2->size[0], m2->size[1]); */

  /* r_ */
  if(r_->stride[0] == 1 &&
     r_->stride[1] != 0)
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if(r_->stride[1] == 1 &&
          r_->stride[0] != 0)
  {
    THTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';

    THTensor *transp_r_ = THTensor_(newTranspose)(r_, 0, 1);
    r__ = THTensor_(newClone)(transp_r_);
    THTensor_(free)(transp_r_);
    THTensor_(transpose)(r__, NULL, 0, 1);
  }

  /* m1 */
  if(m1->stride[(transpose_r == 'n' ? 0 : 1)] == 1 &&
     m1->stride[(transpose_r == 'n' ? 1 : 0)] != 0)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride[(transpose_r == 'n' ? 1 : 0)] == 1 &&
          m1->stride[(transpose_r == 'n' ? 0 : 1)] != 0)
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
    m1_ = THTensor_(newContiguous)(m1);
  }

  /* m2 */
  if(m2->stride[(transpose_r == 'n' ? 0 : 1)] == 1 &&
     m2->stride[(transpose_r == 'n' ? 1 : 0)] != 0)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride[(transpose_r == 'n' ? 1 : 0)] == 1 &&
          m2->stride[(transpose_r == 'n' ? 0 : 1)] != 0)
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
    m2_ = THTensor_(newContiguous)(m2);
  }

  /* do the operation */
  THBlas_(gemm)(transpose_m1,
                transpose_m2,
                r__->size[(transpose_r == 'n' ? 0 : 1)],
                r__->size[(transpose_r == 'n' ? 1 : 0)],
                m1_->size[(transpose_r == 'n' ? 1 : 0)],
                alpha,
                THTensor_(data)(m1_),
                (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                THTensor_(data)(m2_),
                (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                beta,
                THTensor_(data)(r__),
                r__->stride[(transpose_r == 'n' ? 1 : 0)]);

  /* free intermediate variables */
  if(m1_ != m1)
    THTensor_(free)(m1_);

  if(m2_ != m2)
    THTensor_(free)(m2_);

  if(r__ != r_)
    THTensor_(freeCopyTo)(r__, r_);
}


void THTensor_(remainder)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
      real *tp = THTensor_(data)(t);
      real *rp = THTensor_(data)(r_);
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++)
          rp[i] = (value == 0)? NAN : tp[i] - value * floor(tp[i] / value);
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = (value == 0)? NAN : *t_data - value * floor(*t_data / value););
  }
}


void THTensor_(cremainder)(THTensor *r_, THTensor *t, THTensor *src)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(isContiguous)(src) && THTensor_(nElement)(r_) == THTensor_(nElement)(src)) {
      real *tp = THTensor_(data)(t);
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(r_);
      long sz = THTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
      for (i=0; i<sz; i++)
          rp[i] = (sp[i] == 0)? NAN : tp[i] - sp[i] * floor(tp[i] / sp[i]);
  } else {
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = (*src_data == 0)? NAN : *t_data - *src_data * floor(*t_data / *src_data););
  }
}


#endif