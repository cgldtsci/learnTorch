#ifndef TH_ATOMIC_INC
#define TH_ATOMIC_INC

#include "THGeneral.h"

/*
 * return *a
*/
TH_API int THAtomicGet(int volatile *a);

/*
 * *a += value,
 * return previous *a
*/
TH_API int THAtomicAdd(int volatile *a, int value);

/*
 * check if (*a == oldvalue)
 * if true: set *a to newvalue, return 1
 * if false: return 0
*/
TH_API int THAtomicCompareAndSwap(int volatile *a, int oldvalue, int newvalue);


/******************************************************************************
 * refcounting functions
 ******************************************************************************/

/*
 * *a++
*/
TH_API void THAtomicIncrementRef(int volatile *a);

/*
 * *a--,
 * return 1 if *a == 0 after the operation, 0 otherwise
*/
TH_API int THAtomicDecrementRef(int volatile *a);
#endif


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


