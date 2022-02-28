#include "THAtomic.h"
#include <stdio.h>

/*
  Note: I thank Leon Bottou for his useful comments.
  Ronan.
*/

#if defined(USE_C11_ATOMICS)
#include <stdatomic.h>
#endif

#if defined(USE_MSC_ATOMICS)
#include <intrin.h>
#endif

#if !defined(USE_MSC_ATOMICS) && !defined(USE_GCC_ATOMICS) && defined(USE_PTHREAD_ATOMICS)
#include <pthread.h>
static pthread_mutex_t ptm = PTHREAD_MUTEX_INITIALIZER;
#endif

long THAtomicAddLong(long volatile *a, long value)
{
#if defined(USE_C11_ATOMICS)
  return atomic_fetch_add(a, value);
#elif defined(USE_MSC_ATOMICS)
  return _InterlockedExchangeAdd(a, value);
#elif defined(USE_GCC_ATOMICS)
  return __sync_fetch_and_add(a, value);
#else
  long oldvalue;
  do {
    oldvalue = *a;
  } while (!THAtomicCompareAndSwapLong(a, oldvalue, (oldvalue + value)));
  return oldvalue;
#endif
}

int THAtomicAdd(int volatile *a, int value)
{
#if defined(USE_C11_ATOMICS)
  return atomic_fetch_add(a, value);
#elif defined(USE_MSC_ATOMICS)
  return _InterlockedExchangeAdd((long*)a, value);
#elif defined(USE_GCC_ATOMICS)
  return __sync_fetch_and_add(a, value);
#else
  int oldvalue;
  do {
    oldvalue = *a;
  } while (!THAtomicCompareAndSwap(a, oldvalue, (oldvalue + value)));
  return oldvalue;
#endif
}


int THAtomicDecrementRef(int volatile *a)
{
  return (THAtomicAdd(a, -1) == 1);
}

long THAtomicCompareAndSwapLong(long volatile *a, long oldvalue, long newvalue)
{
#if defined(USE_C11_ATOMICS)
  return atomic_compare_exchange_strong(a, &oldvalue, newvalue);
#elif defined(USE_MSC_ATOMICS)
  return (_InterlockedCompareExchange(a, newvalue, oldvalue) == oldvalue);
#elif defined(USE_GCC_ATOMICS)
  return __sync_bool_compare_and_swap(a, oldvalue, newvalue);
#elif defined(USE_PTHREAD_ATOMICS)
  long ret = 0;
  pthread_mutex_lock(&ptm);
  if(*a == oldvalue) {
    *a = newvalue;
    ret = 1;
  }
  pthread_mutex_unlock(&ptm);
  return ret;
#else
#warning THAtomic is not thread safe
  if(*a == oldvalue) {
    *a = newvalue;
    return 1;
  }
  else
    return 0;
#endif
}