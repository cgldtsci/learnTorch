#ifndef TH_ATOMIC_INC
#define TH_ATOMIC_INC

#include "THGeneral.h"

/*
 * *a += value,
 * return previous *a
*/
TH_API long THAtomicAddLong(long volatile *a, long value);

/*
 * check if (*a == oldvalue)
 * if true: set *a to newvalue, return 1
 * if false: return 0
*/
TH_API long THAtomicCompareAndSwapLong(long volatile *a, long oldvalue, long newvalue);

#endif
