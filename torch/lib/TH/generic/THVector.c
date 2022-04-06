#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THVector.c"
#else

static TH_INLINE void THVector_(fill)(real *x, const real c, const long n) {
  long i = 0;

  for(; i < n-4; i += 4)
  {
    x[i] = c;
    x[i+1] = c;
    x[i+2] = c;
    x[i+3] = c;
  }

  for(; i < n; i++)
    x[i] = c;
}

#endif
