#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THBlas.c"
#else

#ifdef BLAS_F2C
# define ffloat double
#else
# define ffloat float
#endif



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

#endif
