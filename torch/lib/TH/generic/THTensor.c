#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensor.c"
#else

/**** creation methods ****/

static void THTensor_(rawInit)(THTensor *self);
static void THTensor_(rawSet)(THTensor *self, THStorage *storage, long storageOffset, int nDimension, long *size, long *stride);
static void THTensor_(rawResize)(THTensor *self, int nDimension, long *size, long *stride);

/* Empty init */
THTensor *THTensor_(new)(void)
{
  THTensor *self = THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  return self;
}

/* Pointer-copy init */
THTensor *THTensor_(newWithTensor)(THTensor *tensor)
{
  THTensor *self = THAlloc(sizeof(THTensor));
  THTensor_(rawInit)(self);
  THTensor_(rawSet)(self,
                    tensor->storage,
                    tensor->storageOffset,
                    tensor->nDimension,
                    tensor->size,
                    tensor->stride);
  return self;
}

/* Storage init */
THTensor *THTensor_(newWithStorage)(THStorage *storage, long storageOffset, THLongStorage *size, THLongStorage *stride)
{
  THTensor *self = THAlloc(sizeof(THTensor));
  if(size && stride)
    THArgCheck(size->size == stride->size, 4, "inconsistent size");

  THTensor_(rawInit)(self);
  THTensor_(rawSet)(self,
                    storage,
                    storageOffset,
                    (size ? size->size : (stride ? stride->size : 0)),
                    (size ? size->data : NULL),
                    (stride ? stride->data : NULL));

  return self;
}

THTensor *THTensor_(newWithSize)(THLongStorage *size, THLongStorage *stride)
{
  return THTensor_(newWithStorage)(NULL, 0, size, stride);
}

void THTensor_(free)(THTensor *self)
{
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED)
  {
    if(THAtomicDecrementRef(&self->refcount))
    {
      THFree(self->size);
      THFree(self->stride);
      if(self->storage)
        THStorage_(free)(self->storage);
      THFree(self);
    }
  }
}


/*******************************************************************************/

static void THTensor_(rawInit)(THTensor *self)
{
  self->refcount = 1;
  self->storage = NULL;
  self->storageOffset = 0;
  self->size = NULL;
  self->stride = NULL;
  self->nDimension = 0;
  self->flag = TH_TENSOR_REFCOUNTED;
}


static void THTensor_(rawSet)(THTensor *self, THStorage *storage, long storageOffset, int nDimension, long *size, long *stride)
{
  /* storage */
  if(self->storage != storage)
  {
    if(self->storage)
      THStorage_(free)(self->storage);

    if(storage)
    {
      self->storage = storage;
      THStorage_(retain)(self->storage);
    }
    else
      self->storage = NULL;
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THTensor_(rawResize)(self, nDimension, size, stride);
}


static void THTensor_(rawResize)(THTensor *self, int nDimension, long *size, long *stride)
{
  int d;
  int nDimension_;
  long totalSize;
  int hascorrectsize = 1;

  nDimension_ = 0;
  for(d = 0; d < nDimension; d++)
  {
    if(size[d] > 0)
    {
      nDimension_++;
      if((self->nDimension > d) && (size[d] != self->size[d]))
        hascorrectsize = 0;

      if((self->nDimension > d) && stride && (stride[d] >= 0) && (stride[d] != self->stride[d]))
        hascorrectsize = 0;
    }
    else
      break;
  }
  nDimension = nDimension_;

  if(nDimension != self->nDimension)
    hascorrectsize = 0;

  if(hascorrectsize)
    return;

  if(nDimension > 0)
  {
    if(nDimension != self->nDimension)
    {
      self->size = THRealloc(self->size, sizeof(long)*nDimension);
      self->stride = THRealloc(self->stride, sizeof(long)*nDimension);
      self->nDimension = nDimension;
    }

    totalSize = 1;
    for(d = self->nDimension-1; d >= 0; d--)
    {
      self->size[d] = size[d];
      if(stride && (stride[d] >= 0) )
        self->stride[d] = stride[d];
      else
      {
        if(d == self->nDimension-1)
          self->stride[d] = 1;
        else
          self->stride[d] = self->size[d+1]*self->stride[d+1];
      }
      totalSize += (self->size[d]-1)*self->stride[d];
    }

    if(totalSize+self->storageOffset > 0)
    {
      if(!self->storage)
        self->storage = THStorage_(new)();
      if(totalSize+self->storageOffset > self->storage->size)
        THStorage_(resize)(self->storage, totalSize+self->storageOffset);
    }
  }
  else
    self->nDimension = 0;
}

#endif