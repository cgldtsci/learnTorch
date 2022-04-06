#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THBlas.c"
#else

#ifdef BLAS_F2C
# define ffloat double
#else
# define ffloat float
#endif

void THBlas_(scal)(long n, real a, real *x, long incx)
{
  if(n == 1)
    incx = 1;

  {
    long i;
    for(i = 0; i < n; i++)
      x[i*incx] *= a;
  }
}

void THBlas_(axpy)(long n, real a, real *x, long incx, real *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

  {
    long i;
    for(i = 0; i < n; i++)
      y[i*incy] += a*x[i*incx];
  }
}


void THBlas_(gemv)(char trans, long m, long n, real alpha, real *a, long lda, real *x, long incx, real beta, real *y, long incy)
{
  if(n == 1)
    lda = m;
  {
    long i, j;

    if( (trans == 'T') || (trans == 't') )
    {
      for(i = 0; i < n; i++)
      {
        real sum = 0;
        real *row_ = a+lda*i;
        for(j = 0; j < m; j++)
          sum += x[j*incx]*row_[j];
	if (beta == 0)
	  y[i*incy] = alpha*sum;
	else
	  y[i*incy] = beta*y[i*incy] + alpha*sum;
      }
    }
    else
    {
      if(beta != 1)
        THBlas_(scal)(m, beta, y, incy);

      for(j = 0; j < n; j++)
      {
        real *column_ = a+lda*j;
        real z = alpha*x[j*incx];
        for(i = 0; i < m; i++)
          y[i*incy] += z*column_[i];
      }
    }
  }
}


void THBlas_(gemm)(char transa, char transb, long m, long n, long k, real alpha, real *a, long lda, real *b, long ldb, real beta, real *c, long ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    ldc = m;

  if(transa_)
  {
    if(m == 1)
      lda = k;
  }
  else
  {
    if(k == 1)
      lda = m;
  }

  if(transb_)
  {
    if(k == 1)
      ldb = n;
  }
  else
  {
    if(n == 1)
      ldb = k;
  }

  {
    long i, j, l;
    if(!transa_ && !transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l];
          b_ += ldb;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else if(transa_ && !transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l];
          b_ += ldb;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
    else if(!transa_ && transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l*ldb];
          b_++;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l*ldb];
          b_++;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
  }
}

#endif
