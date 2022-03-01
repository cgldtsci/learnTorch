#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorage.c"
#else


void THStorage_(retain)(THStorage *storage)
{
  if(storage && (storage->flag & TH_STORAGE_REFCOUNTED))
    THAtomicIncrementRef(&storage->refcount);
}

void THStorage_(free)(THStorage *storage)
{
  if(!storage)
    return;

  if((storage->flag & TH_STORAGE_REFCOUNTED) && (THAtomicGet(&storage->refcount) > 0))
  {
    if(THAtomicDecrementRef(&storage->refcount))
    {
      if(storage->flag & TH_STORAGE_FREEMEM) {
        storage->allocator->free(storage->allocatorContext, storage->data);
      }
      if(storage->flag & TH_STORAGE_VIEW) {
        THStorage_(free)(storage->view);
      }
      THFree(storage);
    }
  }
}

THStorage* THStorage_(new)(void)
{
  return THStorage_(newWithSize)(0);
}

THStorage* THStorage_(newWithSize)(long size)
{
  return THStorage_(newWithAllocator)(size, &THDefaultAllocator, NULL);
}

THStorage* THStorage_(newWithData)(real *data, long size)
{
  return THStorage_(newWithDataAndAllocator)(data, size,
                                             &THDefaultAllocator, NULL);
}


THStorage* THStorage_(newWithDataAndAllocator)(real* data, long size,
                                               THAllocator* allocator,
                                               void* allocatorContext) {
  THStorage *storage = THAlloc(sizeof(THStorage));
  storage->data = data;
  storage->size = size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  storage->allocator = allocator;
  storage->allocatorContext = allocatorContext;
  return storage;
}

#endif
