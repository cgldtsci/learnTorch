#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THBlas.h"
#else

/* Level 1 */
TH_API void THBlas_(axpy)(long n, real a, real *x, long incx, real *y, long incy);

/* Level 2 */
TH_API void THBlas_(gemv)(char trans, long m, long n, real alpha, real *a, long lda, real *x, long incx, real beta, real *y, long incy);

/* Level 3 */
TH_API void THBlas_(gemm)(char transa, char transb, long m, long n, long k, real alpha, real *a, long lda, real *b, long ldb, real beta, real *c, long ldc);

#endif
