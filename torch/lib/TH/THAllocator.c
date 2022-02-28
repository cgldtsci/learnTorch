#include "THAllocator.h"

/* stuff for mapped files */
#ifdef _WIN32
#include <windows.h>
#endif

static void *THDefaultAllocator_alloc(void* ctx, long size) {
  return THAlloc(size);
}

static void *THDefaultAllocator_realloc(void* ctx, void* ptr, long size) {
  return THRealloc(ptr, size);
}

static void THDefaultAllocator_free(void* ctx, void* ptr) {
  THFree(ptr);
}

THAllocator THDefaultAllocator = {
  &THDefaultAllocator_alloc,
  &THDefaultAllocator_realloc,
  &THDefaultAllocator_free
};
