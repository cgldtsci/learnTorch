#ifndef TH_ALLOCATOR_INC
#define TH_ALLOCATOR_INC

#include "THGeneral.h"

/* Custom allocator
 */
typedef struct THAllocator {
  void* (*malloc)(void*, long);
  void* (*realloc)(void*, void*, long);
  void (*free)(void*, void*);
} THAllocator;

/* default malloc/free allocator. malloc and realloc raise an error (using
 * THError) on allocation failure.
 */
extern THAllocator THDefaultAllocator;

#endif
