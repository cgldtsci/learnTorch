
#ifdef THC_GENERIC_FILE
#define THCP_AUTO_GPU 1
#else
#define THCP_AUTO_GPU 0
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
#define RealStr "float"
#else
#define RealStr "int"
#endif

#ifdef THC_REAL_IS_HALF
#define AS_REAL(x) THC_float2half(x)
#else
#define AS_REAL(x) x
#endif

#ifndef THC_GENERIC_FILE
#define IS_CUDA false
#define CUDA_FLOAT false
#else
#define IS_CUDA true
#define CUDA_FLOAT defined(THC_REAL_IS_FLOAT)
#endif

#if IS_CUDA
#define THPIndexTensor THCPLongTensor
#define THPIndexTensorClass THCPLongTensorClass
#else
#define THPIndexTensor THPLongTensor
#define THPIndexTensorClass THPLongTensorClass
#endif

#if IS_CUDA
#define THPBoolTensor THCPByteTensor
#define THPBoolTensorClass THCPByteTensorClass
#else
#define THPBoolTensor THPByteTensor
#define THPBoolTensorClass THPByteTensorClass
#endif

PyObject * THPTensor_(writeMetadata)(THPTensor *self, PyObject *args)
{
  if (!args || PyTuple_Size(args) != 1) {
    THPUtils_invalidArguments(args, "a single file object");
    return NULL;
  }
  int fd = PyObject_AsFileDescriptor(PyTuple_GET_ITEM(args, 0));
  if (fd == -1) {
    THPUtils_setError("write_file couln't retrieve file descriptor from given object");
    return NULL;
  }
  THPTensor_(writeMetadataRaw)(self->cdata, fd);
  Py_RETURN_NONE;
}

PyObject * THPTensor_(newWithMetadataFile)(PyObject *_null, PyObject *args)
{
  if (!args || PyTuple_Size(args) != 2 || !THPStorage_(IsSubclass)(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, "a single file object and a storage object");
    return NULL;
  }
  int fd = PyObject_AsFileDescriptor(PyTuple_GET_ITEM(args, 0));
  if (fd == -1) {
    THPUtils_setError("write_file couln't retrieve file descriptor from given object");
    return NULL;
  }
  THPStorage *storage = (THPStorage*)PyTuple_GET_ITEM(args, 1);
  THTensorPtr tensor = THPTensor_(newWithMetadataFileRaw)(fd, storage->cdata);
  PyObject *result = THPTensor_(newObject)(tensor);
  tensor.release();
  return result;
}


#ifdef NUMPY_TYPE_ENUM
// Adapted from fblualib
PyObject * THPTensor_(toNumpy)(THPTensor *self, PyObject *args) {
  npy_intp zero = 0;
  int ndim;
  npy_intp* sizes_ptr;
  std::unique_ptr<npy_intp[]> sizes;
  std::unique_ptr<npy_intp[]> strides;

  // Numpy and Torch disagree on empty tensors. In Torch, an empty tensor
  // is a tensor with zero dimensions. In Numpy, a tensor with zero dimensions
  // is a scalar (with one element). So we'll convert an empty Torch tensor
  // to a 1d Numpy tensor of shape [0]. Also see pushTensor in PythonToLua.cpp.
  ndim = THTensor_(nDimension)(LIBRARY_STATE self->cdata);
  if (ndim != 0) {

    sizes.reset(new npy_intp[ndim]);
    std::copy(self->cdata->size, self->cdata->size + ndim, sizes.get());
    sizes_ptr = sizes.get();

    if (!THTensor_(isContiguous)(LIBRARY_STATE self->cdata)) {
      strides.reset(new npy_intp[ndim]);
      // Numpy strides use bytes; Torch strides use element counts.
      for (int i = 0; i < ndim; ++i) {
        strides[i] = self->cdata->stride[i] * sizeof(real);
      }
    }
  } else {
    ndim = 1;
    sizes_ptr = &zero;
  }

  THPObjectPtr array = PyArray_New(
      &PyArray_Type, ndim, sizes_ptr, NUMPY_TYPE_ENUM,
      strides.get(), self->cdata->storage->data, 0,
      NPY_ARRAY_ALIGNED, nullptr);
  if (!array) {
    THPUtils_setError("an error occured during conversion to numpy array");
    return NULL;
  }

  // Create a PythonStorage object to hold the reference count.
  // PyArray_SetBaseObject steals the reference to the base object.
  Py_INCREF(self);
  if (PyArray_SetBaseObject((PyArrayObject*)(array.get()), (PyObject*)self) == -1) {
    Py_DECREF(self);
    THPUtils_setError("an error occured during conversion to numpy array");
    return NULL;
  }

  return array.release();
}

THTensor* THPTensor_(fromNumpy)(PyObject *numpy_array) {
  PyArrayObject *array = (PyArrayObject*)numpy_array;
  THStoragePtr storage = THStorage_(newWithDataAndAllocator)(
      (real*)PyArray_DATA(array),
      PyArray_NBYTES(array) / sizeof(real),
      &THNumpyArrayAllocator,
      new NumpyArrayAllocator(numpy_array));

  // Numpy and Torch disagree on empty tensors. In Torch, an empty
  // tensor is a tensor with zero dimensions. In Numpy, an empty tensor
  // keeps its shape, but has 0 as the size of one of the dimensions.
  // So we'll convert all Numpy tensors of 0 elements to empty Torch tensors.
  if (PyArray_SIZE(array) != 0) {
    auto ndim = PyArray_NDIM(array);
    THLongStoragePtr sizes = THLongStorage_newWithSize(ndim);
    long *sizes_data = sizes->data;
    for (int i = 0; i < ndim; ++i) {
      sizes_data[i] = PyArray_DIM(array, i);
    }

    THLongStoragePtr strides = THLongStorage_newWithSize(ndim);
    long *strides_data = strides->data;
    for (int i = 0; i < ndim; ++i) {
      strides_data[i] = PyArray_STRIDE(array, i) / sizeof(real);   // numpy uses bytes, torch uses elements
    }

    THTensor *result = THTensor_(newWithStorage)(storage, 0, sizes, strides);
    // newWithStorage increases refcount
    storage.release();
    return result;
  } else {
    THTensor *result = THTensor_(newWithStorage)(storage, 0, NULL, NULL);
    // newWithStorage increases refcount
    storage.release();
    return result;
  }
}
#endif

#if IS_CUDA
PyObject * THPTensor_(getDevice)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyInt_FromLong(THTensor_(getDevice)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



// TODO: check that there are no args
static PyObject * THPTensor_(elementSize)(THPTensor *self, PyObject *args)
{
  return PyLong_FromLong(THStorage_(elementSize)(LIBRARY_STATE_NOARGS));
}

// TODO: check that there are no args
static PyObject * THPTensor_(storage)(THPTensor *self, PyObject *args)
{
  // TODO: memory leak on error
  THStorage *result = THTensor_(storage)(LIBRARY_STATE self->cdata);
  if (result == NULL)
    Py_RETURN_NONE;
  THStorage_(retain)(LIBRARY_STATE result);
  THStoragePtr _tmp = result;
  PyObject *ret = THPStorage_(newObject)(result);
  _tmp.release();
  return ret;
}

PyObject * THPTensor_(storageOffset)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyInt_FromLong(THTensor_(storageOffset)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(nDimension)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyInt_FromLong(THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(free)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(free)(LIBRARY_STATE ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(retain)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(retain)(LIBRARY_STATE ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(resize_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount > 0) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 0);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(resize)(LIBRARY_STATE ((THPTensor*)self)->cdata, __long_args, NULL);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_stateless_(zeros)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount > 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 1);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(zeros)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, __long_args);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount > 0) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 0);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(zeros)(LIBRARY_STATE ((THPTensor*)result)->cdata, __long_args);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(zeros_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount > 0) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 0);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(zeros)(LIBRARY_STATE ((THPTensor*)self)->cdata, __long_args);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_stateless_(ones)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount > 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 1);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(ones)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, __long_args);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount > 0) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 0);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(ones)(LIBRARY_STATE ((THPTensor*)result)->cdata, __long_args);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(ones_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount > 0) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 0);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(ones)(LIBRARY_STATE ((THPTensor*)self)->cdata, __long_args);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(numel)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyInt_FromLong(THTensor_(numel)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(numel)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyInt_FromLong(THTensor_(numel)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(set_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPStorageClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPLongStorageClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPLongStorageClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(setStorage)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPStorage*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), ((THPLongStorage*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPLongStorage*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount > 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPStorageClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 2);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(setStorage)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPStorage*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), __long_args, NULL);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(set)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPStorageClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THLongStoragePtr __storage_size = THLongStorage_newWithSize1(THStorage_(size)(LIBRARY_STATE ((THPStorage*)PyTuple_GET_ITEM(args, 0))->cdata));THTensor_(setStorage)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPStorage*)PyTuple_GET_ITEM(args, 0))->cdata, 0, __storage_size.get(), NULL);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(setStorage)(LIBRARY_STATE ((THPTensor*)self)->cdata, NULL, 0, NULL, NULL);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(THStorage* sourceStorage, long storageOffset, THLongStorage* sizes, THLongStorage* strides) or (THStorage* sourceStorage, long storageOffset) or (THTensor* source) or (THStorage* storage) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


static PyObject * THPTensor_(select)(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  long dim, idx;
  if (!PyArg_ParseTuple(args, "ll", &dim, &idx))
    return NULL;

  int ndim = THTensor_(nDimension)(LIBRARY_STATE self->cdata);
  if(ndim > 1) {
    THTensor *selected = THTensor_(newWithTensor)(LIBRARY_STATE self->cdata);
    THTensor_(select)(LIBRARY_STATE selected, NULL, dim, idx);
    return THPTensor_(newObject)(selected);
  }
  else {
    THArgCheck(ndim == 1, 1, "empty Tensor");
    return THPUtils_(newReal)(THTensor_(get1d)(LIBRARY_STATE self->cdata, idx));
  }
  END_HANDLE_TH_ERRORS
}

#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
#define BUILD_REAL_FMT "d"
#else
#define BUILD_REAL_FMT "L"
#endif

#if !IS_CUDA
static PyObject * THPTensor_(apply)(THPTensor *self, PyObject *arg)
{
  HANDLE_TH_ERRORS
  if (!PyCallable_Check(arg)) {
    THPUtils_setError("apply requires a callable as it's first argument");
    return NULL;
  }

  real v;
  THTensor *tensor = self->cdata;
  TH_TENSOR_APPLY(real, tensor,
                  PyObject *ret =
                      PyObject_CallFunction(arg, (char*)BUILD_REAL_FMT, *tensor_data);
                  if (!ret)
                    return NULL;
                  bool success = THPUtils_(parseReal)(ret, &v);
                  Py_DECREF(ret);
                  if (!success)
                    THError("given function should return a number");
                  *tensor_data = v;
                  );

  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_(map)(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
    PyObject *fn;
    THPTensor *src_object;
    if (!PyArg_ParseTuple(args, "O!O&", &THPTensorType, &src_object, THPUtils_getCallable, &fn))
      return NULL;

  real v;
  THTensor *tensor = self->cdata;
  THTensor *src = src_object->cdata;
  TH_TENSOR_APPLY2(real, tensor, real, src,
                  PyObject *ret =
                      PyObject_CallFunction(fn, (char*)(BUILD_REAL_FMT BUILD_REAL_FMT),
                                            *tensor_data, *src_data);
                  if (!ret)
                    return NULL;
                  bool success = THPUtils_(parseReal)(ret, &v);
                  Py_DECREF(ret);
                  if (!success)
                    THError("given function should return a number");
                  *tensor_data = v;
                  );

  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_(map2)(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
    PyObject *fn;
    THPTensor *src1_object;
    THPTensor *src2_object;
    if (!PyArg_ParseTuple(args, "O!O!O&", &THPTensorType, &src1_object, &THPTensorType, &src2_object, THPUtils_getCallable, &fn))
      return NULL;

  real v;
  THTensor *tensor = self->cdata;
  THTensor *src1 = src1_object->cdata;
  THTensor *src2 = src2_object->cdata;
  TH_TENSOR_APPLY3(real, tensor, real, src1, real, src2,
                  PyObject *ret =
                      PyObject_CallFunction(fn, (char*)(BUILD_REAL_FMT BUILD_REAL_FMT BUILD_REAL_FMT),
                                            *tensor_data, *src1_data, *src2_data);
                  if (!ret)
                    return NULL;
                  bool success = THPUtils_(parseReal)(ret, &v);
                  Py_DECREF(ret);
                  if (!success)
                    THError("given function should return a number");
                  *tensor_data = v;
                  );

  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}
#endif /* !IS_CUDA */

#undef BUILD_REAL_FMT

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_LONG) || defined(TH_REAL_IS_INT) || CUDA_FLOAT
PyObject * THPTensor_(abs)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(abs)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_LONG) || defined(TH_REAL_IS_INT) || CUDA_FLOAT
PyObject * THPTensor_stateless_(abs)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(abs)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(abs)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_LONG) || defined(TH_REAL_IS_INT) || CUDA_FLOAT
PyObject * THPTensor_(abs_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(abs)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(sigmoid_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sigmoid)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(sigmoid)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(sigmoid)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(sigmoid)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sigmoid)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(sigmoid)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(log_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(log)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(log)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(log)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(log)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(log)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(log)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(log1p_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(log1p)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(log1p)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(log1p)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(log1p)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(log1p)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(log1p)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(exp_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(exp)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(exp)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(exp)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(exp)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(exp)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(exp)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(cos_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cos)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(cos)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cos)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(cos)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cos)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cos)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(acos_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(acos)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(acos)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(acos)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(acos)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(acos)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(acos)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(cosh_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cosh)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(cosh)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cosh)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(cosh)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cosh)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cosh)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(sin_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sin)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(sin)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(sin)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(sin)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sin)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(sin)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(asin_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(asin)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(asin)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(asin)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(asin)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(asin)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(asin)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(sinh_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sinh)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(sinh)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(sinh)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(sinh)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sinh)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(sinh)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(tan_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(tan)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(tan)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(tan)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(tan)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(tan)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(tan)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(atan_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(atan)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(atan)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(atan)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(atan)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(atan)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(atan)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(tanh_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(tanh)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(tanh)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(tanh)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(tanh)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(tanh)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(tanh)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(sqrt_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sqrt)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(sqrt)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(sqrt)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(sqrt)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sqrt)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(sqrt)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(rsqrt_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(rsqrt)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(rsqrt)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(rsqrt)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(rsqrt)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(rsqrt)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(rsqrt)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(ceil_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(ceil)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(ceil)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(ceil)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(ceil)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(ceil)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(ceil)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(floor_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(floor)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(floor)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(floor)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(floor)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(floor)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(floor)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(round_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(round)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(round)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(round)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(round)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(round)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(round)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(trunc_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(trunc)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(trunc)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(trunc)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(trunc)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(trunc)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(trunc)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(frac_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(frac)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(frac)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(frac)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(frac)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(frac)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(frac)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(mean)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(mean)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(meanall)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(long dim) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(mean)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(mean)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(mean)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(meanall)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source, long dim) or (THTensor* source, long dim) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(var)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(var)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), false);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(varall)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(long dim) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(var)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(var)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), false);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(var)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), false);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(varall)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source, long dim) or (THTensor* source, long dim) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(std)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(std)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), false);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(stdall)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(long dim) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(std)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(std)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), false);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(std)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), false);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(stdall)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source, long dim) or (THTensor* source, long dim) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(norm)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(norm)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(normall)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0))));
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(normall)(LIBRARY_STATE ((THPTensor*)self)->cdata, 2));
    
    } else {
      THPUtils_invalidArguments(args, "(real p, long dim) or (real p) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(norm)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(norm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(norm)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(normall)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1))));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(normall)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, 2));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source, real p, long dim) or (THTensor* source, real p, long dim) or (THTensor* source, real p) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(renorm)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(renorm)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(real p, long dim, real maxnorm)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(renorm)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 4))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(renorm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 4)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(renorm)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 3)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source, real p, long dim, real maxnorm) or (THTensor* source, real p, long dim, real maxnorm)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(renorm_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(renorm)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real p, long dim, real maxnorm)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(dist)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(dist)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1))));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(dist)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, 2));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* other, real p) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(dist)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(dist)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2))));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(dist)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, 2));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* source, THTensor* other, real p) or (THTensor* source, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(cinv)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(cinv)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(cinv)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cinv)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(cinv)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(cinv_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cinv)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(neg)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(neg)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(neg)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(neg)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(neg)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(neg_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(neg)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(atan2)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(atan2)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(atan2)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(atan2)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(atan2)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source, THTensor* other) or (THTensor* source, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(atan2_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(atan2)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


// These options look the same in stateful method - only the first one will
// be available. Still, they differ in torch.pow.
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(pow)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(pow)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(cpow)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(tpow)(LIBRARY_STATE ((THPTensor*)destination)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(real exponent) or (THTensor* exponent) or (real base)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(pow)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(pow)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cpow)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(tpow)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(pow)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(cpow)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(tpow)(LIBRARY_STATE ((THPTensor*)destination)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source, real exponent) or (THTensor* destination, THTensor* source, THTensor* exponent) or (THTensor* destination, real base, THTensor* source) or (THTensor* source, real exponent) or (THTensor* source, THTensor* exponent) or (real base, THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(pow_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(pow)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cpow)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(real exponent) or (THTensor* exponent)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(lerp)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(lerp)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* end, real weight)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_stateless_(lerp)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(lerp)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 3)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(lerp)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source, THTensor* end, real weight) or (THTensor* source, THTensor* end, real weight)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
PyObject * THPTensor_(lerp_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(lerp)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* end, real weight)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
PyObject * THPTensor_stateless_(linspace)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(linspace)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(linspace)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(linspace)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), 100);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(linspace)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), 100);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, real start, real end, long steps) or (real start, real end, long steps) or (THTensor* result, real start, real end) or (real start, real end)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
PyObject * THPTensor_stateless_(logspace)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(logspace)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(logspace)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(logspace)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), 100);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(logspace)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), 100);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, real start, real end, long steps) or (real start, real end, long steps) or (THTensor* result, real start, real end) or (real start, real end)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
PyObject * THPTensor_(histc)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(histc)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 2 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(histc)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), 0);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(histc)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), 0, 0);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(histc)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, 100, 0, 0);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(long bins, real min, real max) or (long bins, real min) or (long bins) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
PyObject * THPTensor_stateless_(histc)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 4))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(histc)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 3)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 4)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(histc)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 3)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(histc)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 3)), 0);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(histc)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), 0);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(histc)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), 0, 0);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(histc)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), 0, 0);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(histc)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, 100, 0, 0);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(histc)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, 100, 0, 0);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source, long bins, real min, real max) or (THTensor* source, long bins, real min, real max) or (THTensor* destination, THTensor* source, long bins, real min) or (THTensor* source, long bins, real min) or (THTensor* destination, THTensor* source, long bins) or (THTensor* source, long bins) or (THTensor* destination, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


PyObject * THPTensor_(zero_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(zero)(LIBRARY_STATE ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(size)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyInt_FromLong(THTensor_(size)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0))));
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return THPLongStorage_newObject(THTensor_(newSizeOf)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(long dim) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(stride)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyInt_FromLong(THTensor_(stride)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0))));
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return THPLongStorage_newObject(THTensor_(newStrideOf)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(long dim) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(fill_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(fill)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real value)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(isSameSizeAs)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyBool_FromLong(THTensor_(isSameSizeAs)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(isContiguous)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyBool_FromLong(THTensor_(isContiguous)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(isSetTo)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyBool_FromLong(THTensor_(isSetTo)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* tensor)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(isSize)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPLongStorageClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyBool_FromLong(THTensor_(isSize)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPLongStorage*)PyTuple_GET_ITEM(args, 0))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THLongStorage* size)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(cmax)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cmax)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cmaxValue)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* other) or (real value)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(cmax)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cmax)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cmaxValue)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cmax)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cmaxValue)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, THTensor* other) or (THTensor* result, THTensor* source, real value) or (THTensor* source, THTensor* other) or (THTensor* source, real value)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(cmax_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cmax)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cmaxValue)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* other) or (real value)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(cmin)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cmin)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cminValue)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* other) or (real value)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(cmin)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cmin)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cminValue)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cmin)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cminValue)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, THTensor* other) or (THTensor* result, THTensor* source, real value) or (THTensor* source, THTensor* other) or (THTensor* source, real value)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(cmin_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cmin)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cminValue)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* other) or (real value)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(sum)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(sum)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(sumall)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(long dim) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(sum)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sum)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(sum)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(sumall)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, long dim) or (THTensor* source, long dim) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(prod)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(prod)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(prodall)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(long dim) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(prod)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(prod)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(prod)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(prodall)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, long dim) or (THTensor* source, long dim) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(cumsum)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cumsum)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(long dim)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(cumsum)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cumsum)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cumsum)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, long dim) or (THTensor* source, long dim)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(cumprod)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cumprod)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(long dim)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(cumprod)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cumprod)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cumprod)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, long dim) or (THTensor* source, long dim)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(sign)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(sign)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(sign)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sign)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(sign)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(sign_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sign)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if !IS_CUDA
PyObject * THPTensor_(trace)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(trace)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if !IS_CUDA
PyObject * THPTensor_stateless_(trace)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(trace)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


PyObject * THPTensor_(add)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cadd)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(add)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cadd)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real value, THTensor* other) or (real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(add)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cadd)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cadd)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(add)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cadd)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(add)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cadd)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, real value, THTensor* other) or (THTensor* source, real value, THTensor* other) or (THTensor* result, THTensor* source, real value) or (THTensor* result, THTensor* source, THTensor* other) or (THTensor* source, real value) or (THTensor* source, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(add_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cadd)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(add)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cadd)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(real value, THTensor* other) or (real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}



PyObject * THPTensor_(sub)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(csub)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(sub)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(csub)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real value, THTensor* other) or (real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(sub)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(csub)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(csub)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sub)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(csub)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(sub)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(csub)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, real value, THTensor* other) or (THTensor* source, real value, THTensor* other) or (THTensor* result, THTensor* source, real value) or (THTensor* result, THTensor* source, THTensor* other) or (THTensor* source, real value) or (THTensor* source, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(sub_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(csub)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sub)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(csub)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(real value, THTensor* other) or (real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}



PyObject * THPTensor_(mul)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(mul)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cmul)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(mul)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(mul)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cmul)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(mul)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cmul)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, real value) or (THTensor* result, THTensor* source, THTensor* other) or (THTensor* source, real value) or (THTensor* source, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(mul_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(mul)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cmul)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}



PyObject * THPTensor_(div)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(div)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cdiv)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(div)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(div)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cdiv)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(div)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cdiv)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, real value) or (THTensor* result, THTensor* source, THTensor* other) or (THTensor* source, real value) or (THTensor* source, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(div_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(div)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cdiv)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}



#if !IS_CUDA
PyObject * THPTensor_(fmod)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(fmod)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cfmod)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if !IS_CUDA
PyObject * THPTensor_stateless_(fmod)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(fmod)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cfmod)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(fmod)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cfmod)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, real value) or (THTensor* result, THTensor* source, THTensor* other) or (THTensor* source, real value) or (THTensor* source, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if !IS_CUDA
PyObject * THPTensor_(fmod_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(fmod)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cfmod)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if !IS_CUDA
PyObject * THPTensor_(remainder)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(remainder)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cremainder)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if !IS_CUDA
PyObject * THPTensor_stateless_(remainder)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(remainder)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cremainder)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(remainder)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(cremainder)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, real value) or (THTensor* result, THTensor* source, THTensor* other) or (THTensor* source, real value) or (THTensor* source, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if !IS_CUDA
PyObject * THPTensor_(remainder_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(remainder)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cremainder)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(clamp)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(clamp)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(real min, real max)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(clamp)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(clamp)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 3)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(clamp)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source, real min, real max) or (THTensor* source, real min, real max)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(clamp_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(clamp)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real min, real max)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(dot)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(dot)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* tensor)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(dot)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyFloat_FromDouble(THTensor_(dot)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* source, THTensor* tensor)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(tril)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(tril)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(tril)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, 0);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(long k) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(tril)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(tril)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(tril)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(tril)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, 0);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(tril)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, 0);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source, long k) or (THTensor* source, long k) or (THTensor* destination, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(tril_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(tril)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(tril)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, 0);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(long k) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(triu)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(triu)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(triu)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, 0);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(long k) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(triu)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(triu)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(triu)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(triu)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, 0);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(triu)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, 0);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source, long k) or (THTensor* source, long k) or (THTensor* destination, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(triu_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(triu)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(triu)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, 0);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(long k) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(cross)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(cross)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(cross)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, -1);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* other, long dim) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(cross)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cross)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(cross)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cross)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, -1);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_destination = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _destination_guard = (THPTensor*)THPTensor_(newObject)(_th_destination.get());
      THPTensor* destination = _destination_guard.get();
      _th_destination.release();
      
      THTensor_(cross)(LIBRARY_STATE ((THPTensor*)destination)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, -1);Py_INCREF(destination);
      return (PyObject*)(destination);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* destination, THTensor* source, THTensor* other, long dim) or (THTensor* source, THTensor* other, long dim) or (THTensor* destination, THTensor* source, THTensor* other) or (THTensor* source, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if !IS_CUDA
PyObject * THPTensor_stateless_(eye)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(eye)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(eye)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(eye)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(eye)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, long n, long m) or (long n, long m) or (THTensor* result, long n) or (long n)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if !IS_CUDA
PyObject * THPTensor_(equal)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyBool_FromLong(THTensor_(equal)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if !IS_CUDA
PyObject * THPTensor_stateless_(equal)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyBool_FromLong(THTensor_(equal)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* source, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



#if !IS_CUDA
PyObject * THPTensor_(diag)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(diag)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(diag)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, 0);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(long diagonal) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if !IS_CUDA
PyObject * THPTensor_stateless_(diag)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(diag)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(diag)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(diag)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, 0);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(diag)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, 0);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, long diagonal) or (THTensor* source, long diagonal) or (THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif



PyObject * THPTensor_(lt)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(ltValue)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(ltTensor)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(lt_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(ltValueT)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(ltTensorT)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_stateless_(lt)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(ltValueT)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(ltTensorT)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(ltValue)(LIBRARY_STATE ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(ltTensor)(LIBRARY_STATE ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(ltValue)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(ltTensor)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* tensor, real value) or (THTensor* result, THTensor* tensor, THTensor* other) or (THBoolTensor* result, THTensor* tensor, real value) or (THBoolTensor* result, THTensor* tensor, THTensor* other) or (THTensor* tensor, real value) or (THTensor* tensor, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}



PyObject * THPTensor_(gt)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(gtValue)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(gtTensor)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(gt_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(gtValueT)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(gtTensorT)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_stateless_(gt)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(gtValueT)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(gtTensorT)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(gtValue)(LIBRARY_STATE ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(gtTensor)(LIBRARY_STATE ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(gtValue)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(gtTensor)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* tensor, real value) or (THTensor* result, THTensor* tensor, THTensor* other) or (THBoolTensor* result, THTensor* tensor, real value) or (THBoolTensor* result, THTensor* tensor, THTensor* other) or (THTensor* tensor, real value) or (THTensor* tensor, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}



PyObject * THPTensor_(le)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(leValue)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(leTensor)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(le_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(leValueT)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(leTensorT)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_stateless_(le)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(leValueT)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(leTensorT)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(leValue)(LIBRARY_STATE ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(leTensor)(LIBRARY_STATE ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(leValue)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(leTensor)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* tensor, real value) or (THTensor* result, THTensor* tensor, THTensor* other) or (THBoolTensor* result, THTensor* tensor, real value) or (THBoolTensor* result, THTensor* tensor, THTensor* other) or (THTensor* tensor, real value) or (THTensor* tensor, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}



PyObject * THPTensor_(ge)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(geValue)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(geTensor)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(ge_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(geValueT)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(geTensorT)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_stateless_(ge)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(geValueT)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(geTensorT)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(geValue)(LIBRARY_STATE ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(geTensor)(LIBRARY_STATE ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(geValue)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(geTensor)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* tensor, real value) or (THTensor* result, THTensor* tensor, THTensor* other) or (THBoolTensor* result, THTensor* tensor, real value) or (THBoolTensor* result, THTensor* tensor, THTensor* other) or (THTensor* tensor, real value) or (THTensor* tensor, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}



PyObject * THPTensor_(eq)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(eqValue)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(eqTensor)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(eq_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(eqValueT)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(eqTensorT)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_stateless_(eq)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(eqValueT)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(eqTensorT)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(eqValue)(LIBRARY_STATE ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(eqTensor)(LIBRARY_STATE ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(eqValue)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(eqTensor)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* tensor, real value) or (THTensor* result, THTensor* tensor, THTensor* other) or (THBoolTensor* result, THTensor* tensor, real value) or (THBoolTensor* result, THTensor* tensor, THTensor* other) or (THTensor* tensor, real value) or (THTensor* tensor, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}



PyObject * THPTensor_(ne)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(neValue)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(neTensor)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(ne_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(neValueT)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(neTensorT)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real value) or (THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_stateless_(ne)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(neValueT)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(neTensorT)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(neValue)(LIBRARY_STATE ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(neTensor)(LIBRARY_STATE ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(neValue)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCByteTensorPtr _t_result = THCudaByteTensor_new(LIBRARY_STATE_NOARGS);
      THCPByteTensorPtr _result_guard = (THCPByteTensor*)THCPByteTensor_newObject(_t_result);
      THCPByteTensor *result = _result_guard.get();
      #else
      THByteTensorPtr _t_result = THByteTensor_new();
      THPByteTensorPtr _result_guard = (THPByteTensor*)THPByteTensor_newObject(_t_result);
      THPByteTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(neTensor)(LIBRARY_STATE ((THPBoolTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* tensor, real value) or (THTensor* result, THTensor* tensor, THTensor* other) or (THBoolTensor* result, THTensor* tensor, real value) or (THBoolTensor* result, THTensor* tensor, THTensor* other) or (THTensor* tensor, real value) or (THTensor* tensor, THTensor* other)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(min)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_min = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _min_guard = (THPTensor*)THPTensor_(newObject)(_th_min.get());
      THPTensor* min = _min_guard.get();
      _th_min.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_min_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _min_indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_min_indices);
      THCPLongTensor *min_indices = _min_indices_guard.get();
      #else
      THLongTensorPtr _t_min_indices = THLongTensor_new();
      THPLongTensorPtr _min_indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_min_indices);
      THPLongTensor *min_indices = _min_indices_guard.get();
      #endif
      _t_min_indices.release();
      
      THTensor_(min)(LIBRARY_STATE ((THPTensor*)min)->cdata, ((THPIndexTensor*)min_indices)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));
      return PyTuple_Pack(2, min, min_indices);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return THPUtils_(newReal)(THTensor_(minall)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(long dim) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(min)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(min)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_min = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _min_guard = (THPTensor*)THPTensor_(newObject)(_th_min.get());
      THPTensor* min = _min_guard.get();
      _th_min.release();
      
      THTensor_(min)(LIBRARY_STATE ((THPTensor*)min)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));
      return PyTuple_Pack(2, min, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_min_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _min_indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_min_indices);
      THCPLongTensor *min_indices = _min_indices_guard.get();
      #else
      THLongTensorPtr _t_min_indices = THLongTensor_new();
      THPLongTensorPtr _min_indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_min_indices);
      THPLongTensor *min_indices = _min_indices_guard.get();
      #endif
      _t_min_indices.release();
      
      THTensor_(min)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)min_indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), min_indices);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_min = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _min_guard = (THPTensor*)THPTensor_(newObject)(_th_min.get());
      THPTensor* min = _min_guard.get();
      _th_min.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_min_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _min_indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_min_indices);
      THCPLongTensor *min_indices = _min_indices_guard.get();
      #else
      THLongTensorPtr _t_min_indices = THLongTensor_new();
      THPLongTensorPtr _min_indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_min_indices);
      THPLongTensor *min_indices = _min_indices_guard.get();
      #endif
      _t_min_indices.release();
      
      THTensor_(min)(LIBRARY_STATE ((THPTensor*)min)->cdata, ((THPIndexTensor*)min_indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));
      return PyTuple_Pack(2, min, min_indices);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return THPUtils_(newReal)(THTensor_(minall)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* min, THIndexTensor* min_indices, THTensor* source, long dim) or (THIndexTensor* min_indices, THTensor* source, long dim) or (THTensor* min, THTensor* source, long dim) or (THTensor* source, long dim) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(max)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_max = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _max_guard = (THPTensor*)THPTensor_(newObject)(_th_max.get());
      THPTensor* max = _max_guard.get();
      _th_max.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_max_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _max_indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_max_indices);
      THCPLongTensor *max_indices = _max_indices_guard.get();
      #else
      THLongTensorPtr _t_max_indices = THLongTensor_new();
      THPLongTensorPtr _max_indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_max_indices);
      THPLongTensor *max_indices = _max_indices_guard.get();
      #endif
      _t_max_indices.release();
      
      THTensor_(max)(LIBRARY_STATE ((THPTensor*)max)->cdata, ((THPIndexTensor*)max_indices)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));
      return PyTuple_Pack(2, max, max_indices);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return THPUtils_(newReal)(THTensor_(maxall)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(long dim) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(max)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(max)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_max = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _max_guard = (THPTensor*)THPTensor_(newObject)(_th_max.get());
      THPTensor* max = _max_guard.get();
      _th_max.release();
      
      THTensor_(max)(LIBRARY_STATE ((THPTensor*)max)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));
      return PyTuple_Pack(2, max, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_max_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _max_indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_max_indices);
      THCPLongTensor *max_indices = _max_indices_guard.get();
      #else
      THLongTensorPtr _t_max_indices = THLongTensor_new();
      THPLongTensorPtr _max_indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_max_indices);
      THPLongTensor *max_indices = _max_indices_guard.get();
      #endif
      _t_max_indices.release();
      
      THTensor_(max)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)max_indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), max_indices);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_max = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _max_guard = (THPTensor*)THPTensor_(newObject)(_th_max.get());
      THPTensor* max = _max_guard.get();
      _th_max.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_max_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _max_indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_max_indices);
      THCPLongTensor *max_indices = _max_indices_guard.get();
      #else
      THLongTensorPtr _t_max_indices = THLongTensor_new();
      THPLongTensorPtr _max_indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_max_indices);
      THPLongTensor *max_indices = _max_indices_guard.get();
      #endif
      _t_max_indices.release();
      
      THTensor_(max)(LIBRARY_STATE ((THPTensor*)max)->cdata, ((THPIndexTensor*)max_indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));
      return PyTuple_Pack(2, max, max_indices);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return THPUtils_(newReal)(THTensor_(maxall)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* max, THIndexTensor* max_indices, THTensor* source, long dim) or (THIndexTensor* max_indices, THTensor* source, long dim) or (THTensor* max, THTensor* source, long dim) or (THTensor* source, long dim) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


#if !IS_CUDA
PyObject * THPTensor_(kthvalue)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(kthvalue)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)self)->cdata)-1;THTensor_(kthvalue)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), __last_dim);
      return PyTuple_Pack(2, values, indices);
    
    } else {
      THPUtils_invalidArguments(args, "(long k, long dim) or (long k)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if !IS_CUDA
PyObject * THPTensor_stateless_(kthvalue)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(kthvalue)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4)));
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata)-1;THTensor_(kthvalue)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), __last_dim);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      THTensor_(kthvalue)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));
      return PyTuple_Pack(2, values, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(kthvalue)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), indices);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(kthvalue)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata)-1;THTensor_(kthvalue)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), __last_dim);
      return PyTuple_Pack(2, values, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata)-1;THTensor_(kthvalue)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), __last_dim);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), indices);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata)-1;THTensor_(kthvalue)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), __last_dim);
      return PyTuple_Pack(2, values, indices);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* values, THIndexTensor* indices, THTensor* source, long k, long dim) or (THTensor* values, THIndexTensor* indices, THTensor* source, long k) or (THIndexTensor* indices, THTensor* source, long k, long dim) or (THTensor* values, THTensor* source, long k, long dim) or (THTensor* source, long k, long dim) or (THIndexTensor* indices, THTensor* source, long k) or (THTensor* values, THTensor* source, long k) or (THTensor* source, long k)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if !IS_CUDA
PyObject * THPTensor_(mode)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(mode)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)self)->cdata)-1;THTensor_(mode)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)self)->cdata, __last_dim);
      return PyTuple_Pack(2, values, indices);
    
    } else {
      THPUtils_invalidArguments(args, "(long dim) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if !IS_CUDA
PyObject * THPTensor_stateless_(mode)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(mode)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata)-1;THTensor_(mode)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, __last_dim);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      THTensor_(mode)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));
      return PyTuple_Pack(2, values, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(mode)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), indices);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(mode)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata)-1;THTensor_(mode)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, __last_dim);
      return PyTuple_Pack(2, values, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata)-1;THTensor_(mode)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, __last_dim);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), indices);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata)-1;THTensor_(mode)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, __last_dim);
      return PyTuple_Pack(2, values, indices);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* values, THIndexTensor* indices, THTensor* source, long dim) or (THTensor* values, THIndexTensor* indices, THTensor* source) or (THIndexTensor* indices, THTensor* source, long dim) or (THTensor* values, THTensor* source, long dim) or (THTensor* source, long dim) or (THIndexTensor* indices, THTensor* source) or (THTensor* values, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if !IS_CUDA
PyObject * THPTensor_(median)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(median)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)self)->cdata)-1;THTensor_(median)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)self)->cdata, __last_dim);
      return PyTuple_Pack(2, values, indices);
    
    } else {
      THPUtils_invalidArguments(args, "(long dim) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if !IS_CUDA
PyObject * THPTensor_stateless_(median)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(median)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata)-1;THTensor_(median)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, __last_dim);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      THTensor_(median)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));
      return PyTuple_Pack(2, values, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(median)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), indices);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(median)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata)-1;THTensor_(median)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, __last_dim);
      return PyTuple_Pack(2, values, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata)-1;THTensor_(median)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, __last_dim);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), indices);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata)-1;THTensor_(median)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, __last_dim);
      return PyTuple_Pack(2, values, indices);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* values, THIndexTensor* indices, THTensor* source, long dim) or (THTensor* values, THIndexTensor* indices, THTensor* source) or (THIndexTensor* indices, THTensor* source, long dim) or (THTensor* values, THTensor* source, long dim) or (THTensor* source, long dim) or (THIndexTensor* indices, THTensor* source) or (THTensor* values, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


PyObject * THPTensor_(sort)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(sort)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(sort)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), false);
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)self)->cdata)-1;THTensor_(sort)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)self)->cdata, __last_dim, false);
      return PyTuple_Pack(2, values, indices);
    
    } else {
      THPUtils_invalidArguments(args, "(long dim, bool descending) or (long dim) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(sort)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sort)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4)));
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(sort)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), false);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      THTensor_(sort)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));
      return PyTuple_Pack(2, values, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(sort)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), indices);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(sort)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata)-1;THTensor_(sort)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, __last_dim, false);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      THTensor_(sort)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), false);
      return PyTuple_Pack(2, values, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(sort)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), false);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), indices);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(sort)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), false);
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata)-1;THTensor_(sort)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, __last_dim, false);
      return PyTuple_Pack(2, values, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata)-1;THTensor_(sort)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, __last_dim, false);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), indices);
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata)-1;THTensor_(sort)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, __last_dim, false);
      return PyTuple_Pack(2, values, indices);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* values, THIndexTensor* indices, THTensor* source, long dim, bool descending) or (THTensor* values, THIndexTensor* indices, THTensor* source, long dim) or (THIndexTensor* indices, THTensor* source, long dim, bool descending) or (THTensor* values, THTensor* source, long dim, bool descending) or (THTensor* source, long dim, bool descending) or (THTensor* values, THIndexTensor* indices, THTensor* source) or (THIndexTensor* indices, THTensor* source, long dim) or (THTensor* values, THTensor* source, long dim) or (THTensor* source, long dim) or (THIndexTensor* indices, THTensor* source) or (THTensor* values, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(topk)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), false);
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 2 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), false, false);
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)self)->cdata)-1;THTensor_(topk)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), __last_dim, false, false);
      return PyTuple_Pack(2, values, indices);
    
    } else {
      THPUtils_invalidArguments(args, "(long k, long dim, bool smallest, bool sorted) or (long k, long dim, bool smallest) or (long k, long dim) or (long k)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(topk)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 7 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 6))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 6)));
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 6 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5)));
      return PyTuple_Pack(2, values, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 6 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5)));
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), indices);
    
    } else if (__argcount == 6 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 5))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 5)), false);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4)));
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4)), false);
      return PyTuple_Pack(2, values, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4)), false);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), indices);
    
    } else if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4)), false, false);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), false);
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata)-1;THTensor_(topk)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), __last_dim, false, false);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), PyTuple_GET_ITEM(args, 1));
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), false, false);
      return PyTuple_Pack(2, values, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), false, false);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), indices);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      THTensor_(topk)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), false, false);
      return PyTuple_Pack(2, values, indices);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata)-1;THTensor_(topk)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), __last_dim, false, false);
      return PyTuple_Pack(2, values, PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata)-1;THTensor_(topk)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), __last_dim, false, false);
      return PyTuple_Pack(2, PyTuple_GET_ITEM(args, 0), indices);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_values = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _values_guard = (THPTensor*)THPTensor_(newObject)(_th_values.get());
      THPTensor* values = _values_guard.get();
      _th_values.release();
      
      
      #if IS_CUDA
      THCLongTensorPtr _t_indices = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _indices_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_indices);
      THCPLongTensor *indices = _indices_guard.get();
      #else
      THLongTensorPtr _t_indices = THLongTensor_new();
      THPLongTensorPtr _indices_guard = (THPLongTensor*)THPLongTensor_newObject(_t_indices);
      THPLongTensor *indices = _indices_guard.get();
      #endif
      _t_indices.release();
      
      long __last_dim = THTensor_(nDimension)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata)-1;THTensor_(topk)(LIBRARY_STATE ((THPTensor*)values)->cdata, ((THPIndexTensor*)indices)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), __last_dim, false, false);
      return PyTuple_Pack(2, values, indices);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* values, THIndexTensor* indices, THTensor* source, long k, long dim, bool smallest, bool sorted) or (THIndexTensor* indices, THTensor* source, long k, long dim, bool smallest, bool sorted) or (THTensor* values, THTensor* source, long k, long dim, bool smallest, bool sorted) or (THTensor* values, THIndexTensor* indices, THTensor* source, long k, long dim, bool smallest) or (THTensor* source, long k, long dim, bool smallest, bool sorted) or (THIndexTensor* indices, THTensor* source, long k, long dim, bool smallest) or (THTensor* values, THTensor* source, long k, long dim, bool smallest) or (THTensor* values, THIndexTensor* indices, THTensor* source, long k, long dim) or (THTensor* source, long k, long dim, bool smallest) or (THTensor* values, THIndexTensor* indices, THTensor* source, long k) or (THIndexTensor* indices, THTensor* source, long k, long dim) or (THTensor* values, THTensor* source, long k, long dim) or (THTensor* source, long k, long dim) or (THIndexTensor* indices, THTensor* source, long k) or (THTensor* values, THTensor* source, long k) or (THTensor* source, long k)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


PyObject * THPTensor_(maskedFill_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(maskedFill)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(THBoolTensor* mask, real value)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(maskedCopy_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(maskedCopy)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(THBoolTensor* mask, THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(maskedSelect)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPBoolTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(maskedSelect)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, ((THPBoolTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THBoolTensor* mask)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(maskedSelect)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPBoolTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(maskedSelect)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPBoolTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPBoolTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(maskedSelect)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPBoolTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, THBoolTensor* mask) or (THTensor* source, THBoolTensor* mask)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


#if defined(TH_REAL_IS_BYTE)
PyObject * THPTensor_(all)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyBool_FromLong(THTensor_(logicalall)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_BYTE)
PyObject * THPTensor_(any)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return PyBool_FromLong(THTensor_(logicalany)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


PyObject * THPTensor_(transpose)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return THPTensor_(newObject)(THTensor_(newTranspose)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1))));
    
    } else {
      THPUtils_invalidArguments(args, "(long dim0, long dim1)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(transpose)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return THPTensor_(newObject)(THTensor_(newTranspose)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2))));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* source, long dim0, long dim1)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(transpose_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(transpose)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(long dim0, long dim1)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(t)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return THPTensor_(newObject)(THTensor_(newTranspose)(LIBRARY_STATE ((THPTensor*)self)->cdata, 0, 1));
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(t)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return THPTensor_(newObject)(THTensor_(newTranspose)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, 0, 1));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(t_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(transpose)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, 0, 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(squeeze)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(squeeze1d)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(squeeze)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(long dim) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(squeeze)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(squeeze1d)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(squeeze1d)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(squeeze)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(squeeze)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, long dim) or (THTensor* source, long dim) or (THTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(squeeze_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(squeeze1d)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(squeeze)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(long dim) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


#if !IS_CUDA
PyObject * THPTensor_(nonzero)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_result = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _result_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_result);
      THCPLongTensor *result = _result_guard.get();
      #else
      THLongTensorPtr _t_result = THLongTensor_new();
      THPLongTensorPtr _result_guard = (THPLongTensor*)THPLongTensor_newObject(_t_result);
      THPLongTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(nonzero)(LIBRARY_STATE ((THPIndexTensor*)result)->cdata, ((THPTensor*)self)->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if !IS_CUDA
PyObject * THPTensor_stateless_(nonzero)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(nonzero)(LIBRARY_STATE ((THPIndexTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      #if IS_CUDA
      THCLongTensorPtr _t_result = THCudaLongTensor_new(LIBRARY_STATE_NOARGS);
      THCPLongTensorPtr _result_guard = (THCPLongTensor*)THCPLongTensor_newObject(_t_result);
      THCPLongTensor *result = _result_guard.get();
      #else
      THLongTensorPtr _t_result = THLongTensor_new();
      THPLongTensorPtr _result_guard = (THPLongTensor*)THPLongTensor_newObject(_t_result);
      THPLongTensor *result = _result_guard.get();
      #endif
      _t_result.release();
      
      THTensor_(nonzero)(LIBRARY_STATE ((THPIndexTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THIndexTensor* result, THTensor* source) or (THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


PyObject * THPTensor_(contiguous)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return THPTensor_(newObject)(THTensor_(newContiguous)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(clone)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      return THPTensor_(newObject)(THTensor_(newClone)(LIBRARY_STATE ((THPTensor*)self)->cdata));
    
    } else {
      THPUtils_invalidArguments(args, "no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(resizeAs_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(resizeAs)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* template)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(indexSelect)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(indexSelect)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(long dim, THIndexTensor* index)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(indexSelect)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPIndexTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(indexSelect)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), ((THPIndexTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPIndexTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(indexSelect)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), ((THPIndexTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, long dim, THIndexTensor* index) or (THTensor* source, long dim, THIndexTensor* index)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(indexCopy_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(indexCopy)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(long dim, THIndexTensor* index, THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(indexCopy_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(indexCopy)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), ((THPIndexTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* source, long dim, THIndexTensor* index, THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(indexAdd_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(indexAdd)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(long dim, THIndexTensor* index, THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(indexAdd_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(indexAdd)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), ((THPIndexTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* source, long dim, THIndexTensor* index, THTensor* source)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


PyObject * THPTensor_(indexFill_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(indexFill)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(long dim, THIndexTensor* index, real value)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(indexFill_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPIndexTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(indexFill)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), ((THPIndexTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 3)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* source, long dim, THIndexTensor* index, real value)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(narrow)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(narrow)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(long dimension, long start, long length)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(unfold)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(unfold)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(long dimension, long size, long step)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


#if !IS_CUDA
PyObject * THPTensor_stateless_(range)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(range)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackAccreal)(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackAccreal)(PyTuple_GET_ITEM(args, 2)), THPUtils_(unpackAccreal)(PyTuple_GET_ITEM(args, 3)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(range)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackAccreal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackAccreal)(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackAccreal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(range)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackAccreal)(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackAccreal)(PyTuple_GET_ITEM(args, 2)), 1);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(range)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackAccreal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackAccreal)(PyTuple_GET_ITEM(args, 1)), 1);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, accreal xmin, accreal xmax, accreal step) or (accreal xmin, accreal xmax, accreal step) or (THTensor* result, accreal xmin, accreal xmax) or (accreal xmin, accreal xmax)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if !IS_CUDA
PyObject * THPTensor_(scatter_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(scatter)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 3 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPIndexTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(scatterFill)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), ((THPIndexTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(long dim, THIndexTensor* index, THTensor* src) or (long dim, THIndexTensor* index, real value)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if !IS_CUDA
PyObject * THPTensor_(gather)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPLongTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THLongStoragePtr _size = THLongTensor_newSizeOf(LIBRARY_STATE ((THPLongTensor*)PyTuple_GET_ITEM(args, 1))->cdata);
      THTensor_(resize)(LIBRARY_STATE ((THPTensor*)result)->cdata, _size, NULL);
      THTensor_(gather)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), ((THPLongTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(long dim, THLongTensor* index)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if !IS_CUDA
PyObject * THPTensor_stateless_(gather)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPLongTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THLongStoragePtr _size = THLongTensor_newSizeOf(LIBRARY_STATE ((THPLongTensor*)PyTuple_GET_ITEM(args, 3))->cdata);
      THTensor_(resize)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, _size, NULL);
      THTensor_(gather)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), ((THPLongTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPLongTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THLongStoragePtr _size = THLongTensor_newSizeOf(LIBRARY_STATE ((THPLongTensor*)PyTuple_GET_ITEM(args, 2))->cdata);
      THTensor_(resize)(LIBRARY_STATE ((THPTensor*)result)->cdata, _size, NULL);
      THTensor_(gather)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), ((THPLongTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, long dim, THLongTensor* index) or (THTensor* source, long dim, THLongTensor* index)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


PyObject * THPTensor_(addmm)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real beta, real alpha, THTensor* mat1, THTensor* mat2) or (real beta, THTensor* mat1, THTensor* mat2) or (real alpha, THTensor* mat1, THTensor* mat2) or (THTensor* mat1, THTensor* mat2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(addmm)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 3)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 5)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 3)), ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 5))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 5 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, real beta, THTensor* source, real alpha, THTensor* mat1, THTensor* mat2) or (real beta, THTensor* source, real alpha, THTensor* mat1, THTensor* mat2) or (THTensor* result, real beta, THTensor* source, THTensor* mat1, THTensor* mat2) or (THTensor* result, THTensor* source, real alpha, THTensor* mat1, THTensor* mat2) or (real beta, THTensor* source, THTensor* mat1, THTensor* mat2) or (THTensor* source, real alpha, THTensor* mat1, THTensor* mat2) or (THTensor* result, THTensor* source, THTensor* mat1, THTensor* mat2) or (THTensor* source, THTensor* mat1, THTensor* mat2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(addmm_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real beta, real alpha, THTensor* mat1, THTensor* mat2) or (real beta, THTensor* mat1, THTensor* mat2) or (real alpha, THTensor* mat1, THTensor* mat2) or (THTensor* mat1, THTensor* mat2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(addmv)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real beta, real alpha, THTensor* mat, THTensor* vec) or (real beta, THTensor* mat, THTensor* vec) or (real alpha, THTensor* mat, THTensor* vec) or (THTensor* mat, THTensor* vec)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(addmv)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 3)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 5)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 3)), ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 5))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 5 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, real beta, THTensor* source, real alpha, THTensor* mat, THTensor* vec) or (real beta, THTensor* source, real alpha, THTensor* mat, THTensor* vec) or (THTensor* result, real beta, THTensor* source, THTensor* mat, THTensor* vec) or (THTensor* result, THTensor* source, real alpha, THTensor* mat, THTensor* vec) or (real beta, THTensor* source, THTensor* mat, THTensor* vec) or (THTensor* source, real alpha, THTensor* mat, THTensor* vec) or (THTensor* result, THTensor* source, THTensor* mat, THTensor* vec) or (THTensor* source, THTensor* mat, THTensor* vec)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(addmv_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real beta, real alpha, THTensor* mat, THTensor* vec) or (real beta, THTensor* mat, THTensor* vec) or (real alpha, THTensor* mat, THTensor* vec) or (THTensor* mat, THTensor* vec)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(addr)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real beta, real alpha, THTensor* vec1, THTensor* vec2) or (real beta, THTensor* vec1, THTensor* vec2) or (real alpha, THTensor* vec1, THTensor* vec2) or (THTensor* vec1, THTensor* vec2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(addr)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 3)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 5)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 3)), ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 5))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 5 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, real beta, THTensor* source, real alpha, THTensor* vec1, THTensor* vec2) or (real beta, THTensor* source, real alpha, THTensor* vec1, THTensor* vec2) or (THTensor* result, real beta, THTensor* source, THTensor* vec1, THTensor* vec2) or (THTensor* result, THTensor* source, real alpha, THTensor* vec1, THTensor* vec2) or (real beta, THTensor* source, THTensor* vec1, THTensor* vec2) or (THTensor* source, real alpha, THTensor* vec1, THTensor* vec2) or (THTensor* result, THTensor* source, THTensor* vec1, THTensor* vec2) or (THTensor* source, THTensor* vec1, THTensor* vec2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(addr_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real beta, real alpha, THTensor* vec1, THTensor* vec2) or (real beta, THTensor* vec1, THTensor* vec2) or (real alpha, THTensor* vec1, THTensor* vec2) or (THTensor* vec1, THTensor* vec2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_stateless_(ger)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      long s1 = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, 0);
      long s2 = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, 0);
      THTensor_(resize2d)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, s1, s2);
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(0), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      long s1 = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, 0);
      long s2 = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, 0);
      THTensor_(resize2d)(LIBRARY_STATE ((THPTensor*)result)->cdata, s1, s2);
      THTensor_(addr)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(0), ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* vec1, THTensor* vec2) or (THTensor* vec1, THTensor* vec2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_stateless_(mv)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      long s = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, 0);
      THTensor_(resize1d)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, s);
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(0), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      long s = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, 0);
      THTensor_(resize1d)(LIBRARY_STATE ((THPTensor*)result)->cdata, s);
      THTensor_(addmv)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(0), ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* mat, THTensor* vec) or (THTensor* mat, THTensor* vec)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_stateless_(mm)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      long s1 = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, 0);
      long s2 = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, 1);
      THTensor_(resize2d)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, s1, s2);
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(0), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      long s1 = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, 0);
      long s2 = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, 1);
      THTensor_(resize2d)(LIBRARY_STATE ((THPTensor*)result)->cdata, s1, s2);
      THTensor_(addmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(0), ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* mat1, THTensor* mat2) or (THTensor* mat1, THTensor* mat2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_stateless_(bmm)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      long s1 = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, 0);
      long s2 = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, 1);
      long s3 = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, 2);
      THTensor_(resize3d)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, s1, s2, s3);
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(0), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      long s1 = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, 0);
      long s2 = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, 1);
      long s3 = THTensor_(size)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, 2);
      THTensor_(resize3d)(LIBRARY_STATE ((THPTensor*)result)->cdata, s1, s2, s3);
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(0), ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* mat1, THTensor* mat2) or (THTensor* mat1, THTensor* mat2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(addbmm)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real beta, real alpha, THTensor* batch1, THTensor* batch2) or (real beta, THTensor* batch1, THTensor* batch2) or (real alpha, THTensor* batch1, THTensor* batch2) or (THTensor* batch1, THTensor* batch2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(addbmm)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 3)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 5)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 3)), ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 5))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 5 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, real beta, THTensor* source, real alpha, THTensor* batch1, THTensor* batch2) or (real beta, THTensor* source, real alpha, THTensor* batch1, THTensor* batch2) or (THTensor* result, real beta, THTensor* source, THTensor* batch1, THTensor* batch2) or (THTensor* result, THTensor* source, real alpha, THTensor* batch1, THTensor* batch2) or (real beta, THTensor* source, THTensor* batch1, THTensor* batch2) or (THTensor* source, real alpha, THTensor* batch1, THTensor* batch2) or (THTensor* result, THTensor* source, THTensor* batch1, THTensor* batch2) or (THTensor* source, THTensor* batch1, THTensor* batch2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(addbmm_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addbmm)(LIBRARY_STATE ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real beta, real alpha, THTensor* batch1, THTensor* batch2) or (real beta, THTensor* batch1, THTensor* batch2) or (real alpha, THTensor* batch1, THTensor* batch2) or (THTensor* batch1, THTensor* batch2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(baddbmm)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real beta, real alpha, THTensor* batch1, THTensor* batch2) or (real beta, THTensor* batch1, THTensor* batch2) or (real alpha, THTensor* batch1, THTensor* batch2) or (THTensor* batch1, THTensor* batch2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}

PyObject * THPTensor_stateless_(baddbmm)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 6 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 3)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 5)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 3)), ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 5))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 5 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)result)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, real beta, THTensor* source, real alpha, THTensor* batch1, THTensor* batch2) or (real beta, THTensor* source, real alpha, THTensor* batch1, THTensor* batch2) or (THTensor* result, real beta, THTensor* source, THTensor* batch1, THTensor* batch2) or (THTensor* result, THTensor* source, real alpha, THTensor* batch1, THTensor* batch2) or (real beta, THTensor* source, THTensor* batch1, THTensor* batch2) or (THTensor* source, real alpha, THTensor* batch1, THTensor* batch2) or (THTensor* result, THTensor* source, THTensor* batch1, THTensor* batch2) or (THTensor* source, THTensor* batch1, THTensor* batch2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


PyObject * THPTensor_(baddbmm_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 4 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(baddbmm)(LIBRARY_STATE ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(real beta, real alpha, THTensor* batch1, THTensor* batch2) or (real beta, THTensor* batch1, THTensor* batch2) or (real alpha, THTensor* batch1, THTensor* batch2) or (THTensor* batch1, THTensor* batch2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(addcmul)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addcmul)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addcmul)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real value, THTensor* tensor1, THTensor* tensor2) or (THTensor* tensor1, THTensor* tensor2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(addcmul)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addcmul)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addcmul)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addcmul)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addcmul)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, real value, THTensor* tensor1, THTensor* tensor2) or (THTensor* source, real value, THTensor* tensor1, THTensor* tensor2) or (THTensor* result, THTensor* source, THTensor* tensor1, THTensor* tensor2) or (THTensor* source, THTensor* tensor1, THTensor* tensor2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(addcmul_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addcmul)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addcmul)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(real value, THTensor* tensor1, THTensor* tensor2) or (THTensor* tensor1, THTensor* tensor2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(addcdiv)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addcdiv)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addcdiv)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(real value, THTensor* tensor1, THTensor* tensor2) or (THTensor* tensor1, THTensor* tensor2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_stateless_(addcdiv)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 4)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addcdiv)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)), ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 4))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addcdiv)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 3)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addcdiv)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 3))->cdata);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(addcdiv)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THTensor* source, real value, THTensor* tensor1, THTensor* tensor2) or (THTensor* source, real value, THTensor* tensor1, THTensor* tensor2) or (THTensor* result, THTensor* source, THTensor* tensor1, THTensor* tensor2) or (THTensor* source, THTensor* tensor1, THTensor* tensor2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT || !IS_CUDA
PyObject * THPTensor_(addcdiv_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addcdiv)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(addcdiv)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPTensor*)self)->cdata, AS_REAL(1), ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata);Py_INCREF(self);
      return (PyObject*)(self);
    
    } else {
      THPUtils_invalidArguments(args, "(real value, THTensor* tensor1, THTensor* tensor2) or (THTensor* tensor1, THTensor* tensor2)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if !IS_CUDA
PyObject * THPTensor_stateless_(randperm)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          Py_TYPE(PyTuple_GET_ITEM(args, 1)) == &THPGeneratorType &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(randperm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(randperm)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(randperm)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPDefaultGenerator->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(randperm)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPDefaultGenerator->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THGenerator* generator, long n) or (THGenerator* generator, long n) or (THTensor* result, long n) or (long n)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if !IS_CUDA
static void THTensor_(random2__)(THTensor *self, THGenerator *gen, long a, long b)
{
  THArgCheck(b >= a, 2, "upper bound must be larger than lower bound");
  TH_TENSOR_APPLY(real, self, *self_data = ((THRandom_random(gen) % (b+1-a)) + a);)
}

static void THTensor_(random1__)(THTensor *self, THGenerator *gen, long b)
{
  THArgCheck(b > 0, 1, "upper bound must be strictly positive");
  TH_TENSOR_APPLY(real, self, *self_data = (THRandom_random(gen) % b + 1);)
}
#endif

#if !IS_CUDA
PyObject * THPTensor_(random_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(random2__)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(random1__)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(random2__)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(random)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(random1__)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(random)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(THGenerator* generator, long from, long to) or (THGenerator* generator, long to) or (long from, long to) or (THGenerator* generator) or (long to) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
PyObject * THPTensor_(multinomial)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THLongTensorPtr _th_result = THLongTensor_new(LIBRARY_STATE_NOARGS);
      THPLongTensorPtr _result_guard = (THPLongTensor*)THPLongTensor_newObject(_th_result.get());
      THPLongTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(multinomial)(LIBRARY_STATE ((THPLongTensor*)result)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THLongTensorPtr _th_result = THLongTensor_new(LIBRARY_STATE_NOARGS);
      THPLongTensorPtr _result_guard = (THPLongTensor*)THPLongTensor_newObject(_th_result.get());
      THPLongTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(multinomial)(LIBRARY_STATE ((THPLongTensor*)result)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), false);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 2 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THLongTensorPtr _th_result = THLongTensor_new(LIBRARY_STATE_NOARGS);
      THPLongTensorPtr _result_guard = (THPLongTensor*)THPLongTensor_newObject(_th_result.get());
      THPLongTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(multinomial)(LIBRARY_STATE ((THPLongTensor*)result)->cdata, THPDefaultGenerator->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THLongTensorPtr _th_result = THLongTensor_new(LIBRARY_STATE_NOARGS);
      THPLongTensorPtr _result_guard = (THPLongTensor*)THPLongTensor_newObject(_th_result.get());
      THPLongTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(multinomial)(LIBRARY_STATE ((THPLongTensor*)result)->cdata, THPDefaultGenerator->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), false);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THGenerator* generator, long num_samples, bool replacement) or (THGenerator* generator, long num_samples) or (long num_samples, bool replacement) or (long num_samples)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
PyObject * THPTensor_stateless_(multinomial)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 5 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPLongTensorClass &&
          Py_TYPE(PyTuple_GET_ITEM(args, 1)) == &THPGeneratorType &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(multinomial)(LIBRARY_STATE ((THPLongTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 4 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THLongTensorPtr _th_result = THLongTensor_new(LIBRARY_STATE_NOARGS);
      THPLongTensorPtr _result_guard = (THPLongTensor*)THPLongTensor_newObject(_th_result.get());
      THPLongTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(multinomial)(LIBRARY_STATE ((THPLongTensor*)result)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPLongTensorClass &&
          Py_TYPE(PyTuple_GET_ITEM(args, 1)) == &THPGeneratorType &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 2)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(multinomial)(LIBRARY_STATE ((THPLongTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 1))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 2))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)), false);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 4 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPLongTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 3))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(multinomial)(LIBRARY_STATE ((THPLongTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPDefaultGenerator->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 3)));Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 3 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THLongTensorPtr _th_result = THLongTensor_new(LIBRARY_STATE_NOARGS);
      THPLongTensorPtr _result_guard = (THPLongTensor*)THPLongTensor_newObject(_th_result.get());
      THPLongTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(multinomial)(LIBRARY_STATE ((THPLongTensor*)result)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), false);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THLongTensorPtr _th_result = THLongTensor_new(LIBRARY_STATE_NOARGS);
      THPLongTensorPtr _result_guard = (THPLongTensor*)THPLongTensor_newObject(_th_result.get());
      THPLongTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(multinomial)(LIBRARY_STATE ((THPLongTensor*)result)->cdata, THPDefaultGenerator->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 3 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPLongTensorClass &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(multinomial)(LIBRARY_STATE ((THPLongTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPDefaultGenerator->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 1))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2)), false);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount == 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THLongTensorPtr _th_result = THLongTensor_new(LIBRARY_STATE_NOARGS);
      THPLongTensorPtr _result_guard = (THPLongTensor*)THPLongTensor_newObject(_th_result.get());
      THPLongTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(multinomial)(LIBRARY_STATE ((THPLongTensor*)result)->cdata, THPDefaultGenerator->cdata, ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)), false);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THLongTensor* result, THGenerator* generator, THTensor* source, long num_samples, bool replacement) or (THGenerator* generator, THTensor* source, long num_samples, bool replacement) or (THLongTensor* result, THGenerator* generator, THTensor* source, long num_samples) or (THLongTensor* result, THTensor* source, long num_samples, bool replacement) or (THGenerator* generator, THTensor* source, long num_samples) or (THTensor* source, long num_samples, bool replacement) or (THLongTensor* result, THTensor* source, long num_samples) or (THTensor* source, long num_samples)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
PyObject * THPTensor_(uniform_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(uniform)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(uniform)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(uniform)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(uniform)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, 0, 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(uniform)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(uniform)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, 0, 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(THGenerator* generator, real from, real to) or (THGenerator* generator, real from) or (real from, real to) or (THGenerator* generator) or (real from) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
PyObject * THPTensor_(normal_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(normal)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(normal)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(normal)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(normal)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, 0, 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(normal)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(normal)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, 0, 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(THGenerator* generator, real mean, real var) or (THGenerator* generator, real mean) or (real mean, real var) or (THGenerator* generator) or (real mean) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
PyObject * THPTensor_(cauchy_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cauchy)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cauchy)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cauchy)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cauchy)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, 0, 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cauchy)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cauchy)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, 0, 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(THGenerator* generator, real location, real scale) or (THGenerator* generator, real location) or (real location, real scale) or (THGenerator* generator) or (real location) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
PyObject * THPTensor_(logNormal_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 3 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 2))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(logNormal)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 2)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(logNormal)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)), 2);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(logNormal)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(logNormal)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, 1, 2);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(logNormal)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), 2);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(logNormal)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, 1, 2);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(THGenerator* generator, real location, real scale) or (THGenerator* generator, real location) or (real location, real scale) or (THGenerator* generator) or (real location) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
PyObject * THPTensor_(exponential_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(exponential)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(exponential)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(exponential)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(exponential)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(THGenerator* generator, real lambd) or (THGenerator* generator) or (real lambd) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
PyObject * THPTensor_stateless_(rand)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount > 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          Py_TYPE(PyTuple_GET_ITEM(args, 1)) == &THPGeneratorType) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 2);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(rand)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 1))->cdata, __long_args);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount > 1 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 1);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(rand)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, __long_args);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount > 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 1);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(rand)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPDefaultGenerator->cdata, __long_args);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount > 0) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 0);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(rand)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPDefaultGenerator->cdata, __long_args);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THGenerator* generator) or (THGenerator* generator) or (THTensor* result) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
PyObject * THPTensor_stateless_(randn)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount > 2 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass &&
          Py_TYPE(PyTuple_GET_ITEM(args, 1)) == &THPGeneratorType) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 2);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(randn)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 1))->cdata, __long_args);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount > 1 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 1);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(randn)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, __long_args);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount > 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 1);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(randn)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, THPDefaultGenerator->cdata, __long_args);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount > 0) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 0);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(randn)(LIBRARY_STATE ((THPTensor*)result)->cdata, THPDefaultGenerator->cdata, __long_args);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result, THGenerator* generator) or (THGenerator* generator) or (THTensor* result) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT
PyObject * THPTensor_(multinomial)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(multinomial)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1)));Py_INCREF(result);
      return (PyObject*)(result);
    
    } else if (__argcount == 1 &&
          THPUtils_checkLong(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(multinomial)(LIBRARY_STATE ((THPTensor*)result)->cdata, ((THPTensor*)self)->cdata, THPUtils_unpackLong(PyTuple_GET_ITEM(args, 0)), false);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(long num_samples, bool replacement) or (long num_samples)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT
PyObject * THPTensor_(uniform_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(uniform)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(uniform)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(uniform)(LIBRARY_STATE ((THPTensor*)self)->cdata, 0, 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real from, real to) or (real from) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT
PyObject * THPTensor_(normal_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(normal)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(normal)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(normal)(LIBRARY_STATE ((THPTensor*)self)->cdata, 0, 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real mean, real var) or (real mean) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT
PyObject * THPTensor_(cauchy_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cauchy)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cauchy)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(cauchy)(LIBRARY_STATE ((THPTensor*)self)->cdata, 0, 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real location, real scale) or (real location) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT
PyObject * THPTensor_(logNormal_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0)) &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(logNormal)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(logNormal)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)), 2);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(logNormal)(LIBRARY_STATE ((THPTensor*)self)->cdata, 1, 2);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real location, real scale) or (real location) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT
PyObject * THPTensor_(exponential_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPUtils_(checkReal)(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(exponential)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPUtils_(unpackReal)(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(exponential)(LIBRARY_STATE ((THPTensor*)self)->cdata, 1);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(real lambd) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT
PyObject * THPTensor_stateless_(rand)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount > 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 1);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(rand)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, __long_args);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount > 0) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 0);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(rand)(LIBRARY_STATE ((THPTensor*)result)->cdata, __long_args);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT
PyObject * THPTensor_stateless_(randn)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount > 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPTensorClass) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 1);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(randn)(LIBRARY_STATE ((THPTensor*)PyTuple_GET_ITEM(args, 0))->cdata, __long_args);Py_INCREF(PyTuple_GET_ITEM(args, 0));
      return (PyObject*)(PyTuple_GET_ITEM(args, 0));
    
    } else if (__argcount > 0) {
      THLongStoragePtr __long_args_guard = THPUtils_getLongStorage(args, 0);
      THLongStorage* __long_args = __long_args_guard.get();


      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      THTensorPtr _th_result = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr _result_guard = (THPTensor*)THPTensor_(newObject)(_th_result.get());
      THPTensor* result = _result_guard.get();
      _th_result.release();
      
      THTensor_(randn)(LIBRARY_STATE ((THPTensor*)result)->cdata, __long_args);Py_INCREF(result);
      return (PyObject*)(result);
    
    } else {
      THPUtils_invalidArguments(args, "(THTensor* result) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if !IS_CUDA
PyObject * THPTensor_(geometric_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(geometric)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(geometric)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(THGenerator* generator, double p) or (double p)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if !IS_CUDA
PyObject * THPTensor_(bernoulli_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 2 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 1))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(bernoulli)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 1)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPFloatTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(bernoulli_FloatTensor)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPFloatTensor*)PyTuple_GET_ITEM(args, 1))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 2 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPDoubleTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(bernoulli_DoubleTensor)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, ((THPDoubleTensor*)PyTuple_GET_ITEM(args, 1))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          Py_TYPE(PyTuple_GET_ITEM(args, 0)) == &THPGeneratorType) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(bernoulli)(LIBRARY_STATE ((THPTensor*)self)->cdata, ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata, 0.5);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(bernoulli)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPFloatTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(bernoulli_FloatTensor)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, ((THPFloatTensor*)PyTuple_GET_ITEM(args, 0))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 1 &&
          (PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 0)) == THPDoubleTensorClass) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(bernoulli_DoubleTensor)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, ((THPDoubleTensor*)PyTuple_GET_ITEM(args, 0))->cdata);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(bernoulli)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDefaultGenerator->cdata, 0.5);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(THGenerator* generator, double p) or (THGenerator* generator, THFloatTensor* float_p) or (THGenerator* generator, THDoubleTensor* float_p) or (THGenerator* generator) or (double p) or (THFloatTensor* float_p) or (THDoubleTensor* float_p) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT
PyObject * THPTensor_(geometric_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(geometric)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(double p)");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if CUDA_FLOAT
PyObject * THPTensor_(bernoulli_)(PyObject *self, PyObject *args)
{
    HANDLE_TH_ERRORS
    int __argcount = args ? PyTuple_Size(args) : 0;
    
    if (__argcount == 1 &&
          THPDoubleUtils_checkReal(PyTuple_GET_ITEM(args, 0))) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(bernoulli)(LIBRARY_STATE ((THPTensor*)self)->cdata, THPDoubleUtils_unpackReal(PyTuple_GET_ITEM(args, 0)));
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else if (__argcount == 0) {

      
      #if IS_CUDA
      THCPAutoGPU __autogpu_guard = THCPAutoGPU(args, (PyObject*)self);
      #endif
      
      THTensor_(bernoulli)(LIBRARY_STATE ((THPTensor*)self)->cdata, 0.5);
      Py_INCREF(self);
      return (PyObject*)self;
    
    } else {
      THPUtils_invalidArguments(args, "(double p) or no arguments");
      return NULL;
    }
    END_HANDLE_TH_ERRORS
}
#endif


#if !IS_CUDA || CUDA_FLOAT
static std::pair<std::vector<THPObjectPtr>, std::vector<THTensor *>>
THPTensor_(_iterableTensors)(PyObject *iterable)
{
  THPObjectPtr iterator;
  THPObjectPtr item;
  std::vector<THPObjectPtr> items;
  std::vector<THTensor *> item_tensors;
  if ((iterator = PyObject_GetIter(iterable))) {
    while((item = PyIter_Next(iterator))) {
      if (!THPTensor_(IsSubclass)(item)) {
        THPUtils_setError("expected an iterable of " THPTensorStr ", but found %s in it", Py_TYPE(item)->tp_name);
        throw std::exception();
      }
      item_tensors.push_back(((THPTensor*)item.get())->cdata);
      items.emplace_back(std::move(item));
    }
  } else {
    throw std::invalid_argument("");
  }
  return std::make_pair(std::move(items), std::move(item_tensors));
}

static PyObject * THPTensor_(cat)(THPTensor *self, PyObject *args)
{
#if IS_CUDA && THCP_AUTO_GPU
  THCPAutoGPU __autogpu_guard = THCPAutoGPU(args);
#endif
  HANDLE_TH_ERRORS
  THPTensor *tensor1;
  THPTensor *tensor2;
  long dimension;
  Py_ssize_t _argcount = PyTuple_Size(args);
  if (_argcount == 2) {
      PyObject *iterable = PyTuple_GET_ITEM(args, 0);
      PyObject *dim = PyTuple_GET_ITEM(args, 1);
      std::vector<THPObjectPtr> items;
      std::vector<THTensor *> item_tensors;
      if (THPUtils_getLong(dim, &dimension)) {
        try {
          std::tie(items, item_tensors) = THPTensor_(_iterableTensors)(iterable);
          THTensor_(catArray)(LIBRARY_STATE self->cdata, item_tensors.data(), items.size(), dimension);
          Py_INCREF(self);
          return (PyObject*)self;

        } catch (std::invalid_argument &e) {
        } catch (std::exception &e) {
          return NULL;
        }
      }
  } else if (_argcount == 3) {
    if (PyArg_ParseTuple(args, "O!O!l", &THPTensorType, &tensor1, &THPTensorType, &tensor2, &dimension)) {
      THTensor_(cat)(LIBRARY_STATE self->cdata, tensor1->cdata, tensor2->cdata, dimension);
      Py_INCREF(self);
      return (PyObject*)self;
    }
  }

  // TODO: describe args
  THPUtils_invalidArguments(args, "(TODO)");
  return NULL;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_stateless_(cat)(THPTensor *_unused, PyObject *args)
{
#if IS_CUDA && THCP_AUTO_GPU
  THCPAutoGPU __autogpu_guard = THCPAutoGPU(args);
#endif
  HANDLE_TH_ERRORS
  THPTensor *tensor1;
  THPTensor *tensor2;
  long dimension;
  Py_ssize_t _argcount = PyTuple_Size(args);
  if (_argcount == 2) {
    THTensorPtr _self = THTensor_(new)(LIBRARY_STATE_NOARGS);
    THPTensorPtr self = (THPTensor*)THPTensor_(newObject)(_self);
    _self.release();

    PyObject *iterable = PyTuple_GET_ITEM(args, 0);
    PyObject *dim = PyTuple_GET_ITEM(args, 1);
    std::vector<THPObjectPtr> items;
    std::vector<THTensor *> item_tensors;
    if (THPUtils_getLong(dim, &dimension)) {
      try {
        std::tie(items, item_tensors) = THPTensor_(_iterableTensors)(iterable);
        THTensor_(catArray)(LIBRARY_STATE self->cdata, item_tensors.data(), items.size(), dimension);
        Py_INCREF(self.get());
        return (PyObject*)self.release();

      } catch (std::invalid_argument &e) {
      } catch (std::exception &e) {
        return NULL;
      }
    }
  } else if (_argcount == 3) {
    if (PyArg_ParseTuple(args, "O!O!l", &THPTensorType, &tensor1, &THPTensorType, &tensor2, &dimension)) {
      THTensorPtr _self = THTensor_(new)(LIBRARY_STATE_NOARGS);
      THPTensorPtr self = (THPTensor*)THPTensor_(newObject)(_self.get());
      _self.release();

      THTensor_(cat)(LIBRARY_STATE self->cdata, tensor1->cdata, tensor2->cdata, dimension);
      Py_INCREF(self.get());
      return (PyObject*)self.release();
    } else {
      PyErr_Clear();
      THPTensor *self = (THPTensor*)PyTuple_GET_ITEM(args, 0);
      PyObject *iterable = PyTuple_GET_ITEM(args, 1);
      PyObject *dim = PyTuple_GET_ITEM(args, 2);

      std::vector<THPObjectPtr> items;
      std::vector<THTensor *> item_tensors;
      if (THPUtils_getLong(dim, &dimension)) {
        try {
          std::tie(items, item_tensors) = THPTensor_(_iterableTensors)(iterable);
          THTensor_(catArray)(LIBRARY_STATE self->cdata, item_tensors.data(), items.size(), dimension);
          Py_INCREF(self);
          return (PyObject*)self;

        } catch (std::invalid_argument &e) {
        } catch (std::exception &e) {
          fprintf(stderr, "e: %s\n", e.what());
          return NULL;
        }
      }
    }
  } else if (_argcount == 4) {
    THPTensor *self;
    if (PyArg_ParseTuple(args, "O!O!O!l", &THPTensorType, &self, &THPTensorType, &tensor1, &THPTensorType, &tensor2, &dimension)) {
      THTensor_(cat)(LIBRARY_STATE self->cdata, tensor1->cdata, tensor2->cdata, dimension);
      Py_INCREF(self);
      return (PyObject*)self;
    }
  }

  // TODO: describe args
  THPUtils_invalidArguments(args, "(TODO)");
  return NULL;
  END_HANDLE_TH_ERRORS
}
#endif


// cwrap should put definitions before undefs, so let's mark this place

static PyMethodDef THPTensor_(methods)[] = {
  {"_write_metadata", (PyCFunction)THPTensor_(writeMetadata), METH_VARARGS, NULL},
  {"_new_with_metadata_file", (PyCFunction)THPTensor_(newWithMetadataFile), METH_VARARGS | METH_STATIC, NULL},
#if defined(NUMPY_TYPE_ENUM)
  {"numpy", (PyCFunction)THPTensor_(toNumpy), METH_VARARGS, NULL},
#endif
#if IS_CUDA
  {"getDevice", (PyCFunction)THPTensor_(getDevice), METH_VARARGS, NULL},
#endif
  {"elementSize", (PyCFunction)THPTensor_(elementSize), METH_VARARGS, NULL},
  {"storage", (PyCFunction)THPTensor_(storage), METH_VARARGS, NULL},
  {"storageOffset", (PyCFunction)THPTensor_(storageOffset), METH_VARARGS, NULL},
  {"nDimension", (PyCFunction)THPTensor_(nDimension), METH_VARARGS, NULL},
  {"dim", (PyCFunction)THPTensor_(nDimension), METH_VARARGS, NULL},
  {"free", (PyCFunction)THPTensor_(free), METH_VARARGS, NULL},
  {"retain", (PyCFunction)THPTensor_(retain), METH_VARARGS, NULL},
  {"resize_", (PyCFunction)THPTensor_(resize_), METH_VARARGS, NULL},
  {"zeros_", (PyCFunction)THPTensor_(zeros_), METH_VARARGS, NULL},
  {"ones_", (PyCFunction)THPTensor_(ones_), METH_VARARGS, NULL},
  {"numel", (PyCFunction)THPTensor_(numel), METH_VARARGS, NULL},
  {"nElement", (PyCFunction)THPTensor_(numel), METH_VARARGS, NULL},
  {"set_", (PyCFunction)THPTensor_(set_), METH_VARARGS, NULL},
  {"select", (PyCFunction)THPTensor_(select), METH_VARARGS, NULL},
#if !IS_CUDA
  {"apply_", (PyCFunction)THPTensor_(apply), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"map_", (PyCFunction)THPTensor_(map), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"map2_", (PyCFunction)THPTensor_(map2), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_LONG) || defined(TH_REAL_IS_INT) || CUDA_FLOAT
  {"abs", (PyCFunction)THPTensor_(abs), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_LONG) || defined(TH_REAL_IS_INT) || CUDA_FLOAT
  {"abs_", (PyCFunction)THPTensor_(abs_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"sigmoid_", (PyCFunction)THPTensor_(sigmoid_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"sigmoid", (PyCFunction)THPTensor_(sigmoid), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"log_", (PyCFunction)THPTensor_(log_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"log", (PyCFunction)THPTensor_(log), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"log1p_", (PyCFunction)THPTensor_(log1p_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"log1p", (PyCFunction)THPTensor_(log1p), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"exp_", (PyCFunction)THPTensor_(exp_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"exp", (PyCFunction)THPTensor_(exp), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"cos_", (PyCFunction)THPTensor_(cos_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"cos", (PyCFunction)THPTensor_(cos), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"acos_", (PyCFunction)THPTensor_(acos_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"acos", (PyCFunction)THPTensor_(acos), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"cosh_", (PyCFunction)THPTensor_(cosh_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"cosh", (PyCFunction)THPTensor_(cosh), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"sin_", (PyCFunction)THPTensor_(sin_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"sin", (PyCFunction)THPTensor_(sin), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"asin_", (PyCFunction)THPTensor_(asin_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"asin", (PyCFunction)THPTensor_(asin), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"sinh_", (PyCFunction)THPTensor_(sinh_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"sinh", (PyCFunction)THPTensor_(sinh), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"tan_", (PyCFunction)THPTensor_(tan_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"tan", (PyCFunction)THPTensor_(tan), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"atan_", (PyCFunction)THPTensor_(atan_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"atan", (PyCFunction)THPTensor_(atan), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"tanh_", (PyCFunction)THPTensor_(tanh_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"tanh", (PyCFunction)THPTensor_(tanh), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"sqrt_", (PyCFunction)THPTensor_(sqrt_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"sqrt", (PyCFunction)THPTensor_(sqrt), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"rsqrt_", (PyCFunction)THPTensor_(rsqrt_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"rsqrt", (PyCFunction)THPTensor_(rsqrt), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"ceil_", (PyCFunction)THPTensor_(ceil_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"ceil", (PyCFunction)THPTensor_(ceil), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"floor_", (PyCFunction)THPTensor_(floor_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"floor", (PyCFunction)THPTensor_(floor), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"round_", (PyCFunction)THPTensor_(round_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"round", (PyCFunction)THPTensor_(round), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"trunc_", (PyCFunction)THPTensor_(trunc_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"trunc", (PyCFunction)THPTensor_(trunc), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"frac_", (PyCFunction)THPTensor_(frac_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"frac", (PyCFunction)THPTensor_(frac), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"mean", (PyCFunction)THPTensor_(mean), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"var", (PyCFunction)THPTensor_(var), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"std", (PyCFunction)THPTensor_(std), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"norm", (PyCFunction)THPTensor_(norm), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"renorm", (PyCFunction)THPTensor_(renorm), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"renorm_", (PyCFunction)THPTensor_(renorm_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"dist", (PyCFunction)THPTensor_(dist), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"cinv", (PyCFunction)THPTensor_(cinv), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"cinv_", (PyCFunction)THPTensor_(cinv_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"neg", (PyCFunction)THPTensor_(neg), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"neg_", (PyCFunction)THPTensor_(neg_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"atan2", (PyCFunction)THPTensor_(atan2), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"atan2_", (PyCFunction)THPTensor_(atan2_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"pow", (PyCFunction)THPTensor_(pow), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"pow_", (PyCFunction)THPTensor_(pow_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"lerp", (PyCFunction)THPTensor_(lerp), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"lerp_", (PyCFunction)THPTensor_(lerp_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"histc", (PyCFunction)THPTensor_(histc), METH_VARARGS, NULL},
#endif
  {"zero_", (PyCFunction)THPTensor_(zero_), METH_VARARGS, NULL},
  {"size", (PyCFunction)THPTensor_(size), METH_VARARGS, NULL},
  {"stride", (PyCFunction)THPTensor_(stride), METH_VARARGS, NULL},
  {"fill_", (PyCFunction)THPTensor_(fill_), METH_VARARGS, NULL},
  {"isSameSizeAs", (PyCFunction)THPTensor_(isSameSizeAs), METH_VARARGS, NULL},
  {"isContiguous", (PyCFunction)THPTensor_(isContiguous), METH_VARARGS, NULL},
  {"isSetTo", (PyCFunction)THPTensor_(isSetTo), METH_VARARGS, NULL},
  {"isSize", (PyCFunction)THPTensor_(isSize), METH_VARARGS, NULL},
#if CUDA_FLOAT || !IS_CUDA
  {"cmax", (PyCFunction)THPTensor_(cmax), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"cmax_", (PyCFunction)THPTensor_(cmax_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"cmin", (PyCFunction)THPTensor_(cmin), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"cmin_", (PyCFunction)THPTensor_(cmin_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"sum", (PyCFunction)THPTensor_(sum), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"prod", (PyCFunction)THPTensor_(prod), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"cumsum", (PyCFunction)THPTensor_(cumsum), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"cumprod", (PyCFunction)THPTensor_(cumprod), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"sign", (PyCFunction)THPTensor_(sign), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"sign_", (PyCFunction)THPTensor_(sign_), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"trace", (PyCFunction)THPTensor_(trace), METH_VARARGS, NULL},
#endif
  {"add", (PyCFunction)THPTensor_(add), METH_VARARGS, NULL},
  {"add_", (PyCFunction)THPTensor_(add_), METH_VARARGS, NULL},
  {"sub", (PyCFunction)THPTensor_(sub), METH_VARARGS, NULL},
  {"sub_", (PyCFunction)THPTensor_(sub_), METH_VARARGS, NULL},
  {"mul", (PyCFunction)THPTensor_(mul), METH_VARARGS, NULL},
  {"mul_", (PyCFunction)THPTensor_(mul_), METH_VARARGS, NULL},
  {"div", (PyCFunction)THPTensor_(div), METH_VARARGS, NULL},
  {"div_", (PyCFunction)THPTensor_(div_), METH_VARARGS, NULL},
#if !IS_CUDA
  {"fmod", (PyCFunction)THPTensor_(fmod), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"fmod_", (PyCFunction)THPTensor_(fmod_), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"remainder", (PyCFunction)THPTensor_(remainder), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"remainder_", (PyCFunction)THPTensor_(remainder_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"clamp", (PyCFunction)THPTensor_(clamp), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"clamp_", (PyCFunction)THPTensor_(clamp_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"dot", (PyCFunction)THPTensor_(dot), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"tril", (PyCFunction)THPTensor_(tril), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"tril_", (PyCFunction)THPTensor_(tril_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"triu", (PyCFunction)THPTensor_(triu), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"triu_", (PyCFunction)THPTensor_(triu_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"cross", (PyCFunction)THPTensor_(cross), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"equal", (PyCFunction)THPTensor_(equal), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"diag", (PyCFunction)THPTensor_(diag), METH_VARARGS, NULL},
#endif
  {"lt", (PyCFunction)THPTensor_(lt), METH_VARARGS, NULL},
  {"lt_", (PyCFunction)THPTensor_(lt_), METH_VARARGS, NULL},
  {"gt", (PyCFunction)THPTensor_(gt), METH_VARARGS, NULL},
  {"gt_", (PyCFunction)THPTensor_(gt_), METH_VARARGS, NULL},
  {"le", (PyCFunction)THPTensor_(le), METH_VARARGS, NULL},
  {"le_", (PyCFunction)THPTensor_(le_), METH_VARARGS, NULL},
  {"ge", (PyCFunction)THPTensor_(ge), METH_VARARGS, NULL},
  {"ge_", (PyCFunction)THPTensor_(ge_), METH_VARARGS, NULL},
  {"eq", (PyCFunction)THPTensor_(eq), METH_VARARGS, NULL},
  {"eq_", (PyCFunction)THPTensor_(eq_), METH_VARARGS, NULL},
  {"ne", (PyCFunction)THPTensor_(ne), METH_VARARGS, NULL},
  {"ne_", (PyCFunction)THPTensor_(ne_), METH_VARARGS, NULL},
  {"min", (PyCFunction)THPTensor_(min), METH_VARARGS, NULL},
  {"max", (PyCFunction)THPTensor_(max), METH_VARARGS, NULL},
#if !IS_CUDA
  {"kthvalue", (PyCFunction)THPTensor_(kthvalue), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"mode", (PyCFunction)THPTensor_(mode), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"median", (PyCFunction)THPTensor_(median), METH_VARARGS, NULL},
#endif
  {"sort", (PyCFunction)THPTensor_(sort), METH_VARARGS, NULL},
#if CUDA_FLOAT || !IS_CUDA
  {"topk", (PyCFunction)THPTensor_(topk), METH_VARARGS, NULL},
#endif
  {"maskedFill_", (PyCFunction)THPTensor_(maskedFill_), METH_VARARGS, NULL},
  {"maskedCopy_", (PyCFunction)THPTensor_(maskedCopy_), METH_VARARGS, NULL},
  {"maskedSelect", (PyCFunction)THPTensor_(maskedSelect), METH_VARARGS, NULL},
#if defined(TH_REAL_IS_BYTE)
  {"all", (PyCFunction)THPTensor_(all), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_BYTE)
  {"any", (PyCFunction)THPTensor_(any), METH_VARARGS, NULL},
#endif
  {"transpose", (PyCFunction)THPTensor_(transpose), METH_VARARGS, NULL},
  {"transpose_", (PyCFunction)THPTensor_(transpose_), METH_VARARGS, NULL},
  {"t", (PyCFunction)THPTensor_(t), METH_VARARGS, NULL},
  {"t_", (PyCFunction)THPTensor_(t_), METH_VARARGS, NULL},
  {"squeeze", (PyCFunction)THPTensor_(squeeze), METH_VARARGS, NULL},
  {"squeeze_", (PyCFunction)THPTensor_(squeeze_), METH_VARARGS, NULL},
#if !IS_CUDA
  {"nonzero", (PyCFunction)THPTensor_(nonzero), METH_VARARGS, NULL},
#endif
  {"contiguous", (PyCFunction)THPTensor_(contiguous), METH_VARARGS, NULL},
  {"clone", (PyCFunction)THPTensor_(clone), METH_VARARGS, NULL},
  {"resizeAs_", (PyCFunction)THPTensor_(resizeAs_), METH_VARARGS, NULL},
  {"indexSelect", (PyCFunction)THPTensor_(indexSelect), METH_VARARGS, NULL},
  {"indexCopy_", (PyCFunction)THPTensor_(indexCopy_), METH_VARARGS, NULL},
#if CUDA_FLOAT || !IS_CUDA
  {"indexAdd_", (PyCFunction)THPTensor_(indexAdd_), METH_VARARGS, NULL},
#endif
  {"indexFill_", (PyCFunction)THPTensor_(indexFill_), METH_VARARGS, NULL},
  {"narrow", (PyCFunction)THPTensor_(narrow), METH_VARARGS, NULL},
  {"unfold", (PyCFunction)THPTensor_(unfold), METH_VARARGS, NULL},
#if !IS_CUDA
  {"scatter_", (PyCFunction)THPTensor_(scatter_), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"gather", (PyCFunction)THPTensor_(gather), METH_VARARGS, NULL},
#endif
  {"addmm", (PyCFunction)THPTensor_(addmm), METH_VARARGS, NULL},
  {"addmm_", (PyCFunction)THPTensor_(addmm_), METH_VARARGS, NULL},
  {"addmv", (PyCFunction)THPTensor_(addmv), METH_VARARGS, NULL},
  {"addmv_", (PyCFunction)THPTensor_(addmv_), METH_VARARGS, NULL},
  {"addr", (PyCFunction)THPTensor_(addr), METH_VARARGS, NULL},
  {"addr_", (PyCFunction)THPTensor_(addr_), METH_VARARGS, NULL},
  {"addbmm", (PyCFunction)THPTensor_(addbmm), METH_VARARGS, NULL},
  {"addbmm_", (PyCFunction)THPTensor_(addbmm_), METH_VARARGS, NULL},
  {"baddbmm", (PyCFunction)THPTensor_(baddbmm), METH_VARARGS, NULL},
  {"baddbmm_", (PyCFunction)THPTensor_(baddbmm_), METH_VARARGS, NULL},
#if CUDA_FLOAT || !IS_CUDA
  {"addcmul", (PyCFunction)THPTensor_(addcmul), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"addcmul_", (PyCFunction)THPTensor_(addcmul_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"addcdiv", (PyCFunction)THPTensor_(addcdiv), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"addcdiv_", (PyCFunction)THPTensor_(addcdiv_), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"random_", (PyCFunction)THPTensor_(random_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"multinomial", (PyCFunction)THPTensor_(multinomial), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"uniform_", (PyCFunction)THPTensor_(uniform_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"normal_", (PyCFunction)THPTensor_(normal_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"cauchy_", (PyCFunction)THPTensor_(cauchy_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"logNormal_", (PyCFunction)THPTensor_(logNormal_), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"exponential_", (PyCFunction)THPTensor_(exponential_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT
  {"multinomial", (PyCFunction)THPTensor_(multinomial), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT
  {"uniform_", (PyCFunction)THPTensor_(uniform_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT
  {"normal_", (PyCFunction)THPTensor_(normal_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT
  {"cauchy_", (PyCFunction)THPTensor_(cauchy_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT
  {"logNormal_", (PyCFunction)THPTensor_(logNormal_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT
  {"exponential_", (PyCFunction)THPTensor_(exponential_), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"geometric_", (PyCFunction)THPTensor_(geometric_), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"bernoulli_", (PyCFunction)THPTensor_(bernoulli_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT
  {"geometric_", (PyCFunction)THPTensor_(geometric_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT
  {"bernoulli_", (PyCFunction)THPTensor_(bernoulli_), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"cat", (PyCFunction)THPTensor_(cat), METH_VARARGS, NULL},
#endif

  {NULL}
};

static PyMethodDef THPTensor_stateless_(methods)[] = {
  {"zeros", (PyCFunction)THPTensor_stateless_(zeros), METH_VARARGS, NULL},
  {"ones", (PyCFunction)THPTensor_stateless_(ones), METH_VARARGS, NULL},
  {"numel", (PyCFunction)THPTensor_stateless_(numel), METH_VARARGS, NULL},
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_LONG) || defined(TH_REAL_IS_INT) || CUDA_FLOAT
  {"abs", (PyCFunction)THPTensor_stateless_(abs), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"sigmoid", (PyCFunction)THPTensor_stateless_(sigmoid), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"log", (PyCFunction)THPTensor_stateless_(log), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"log1p", (PyCFunction)THPTensor_stateless_(log1p), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"exp", (PyCFunction)THPTensor_stateless_(exp), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"cos", (PyCFunction)THPTensor_stateless_(cos), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"acos", (PyCFunction)THPTensor_stateless_(acos), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"cosh", (PyCFunction)THPTensor_stateless_(cosh), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"sin", (PyCFunction)THPTensor_stateless_(sin), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"asin", (PyCFunction)THPTensor_stateless_(asin), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"sinh", (PyCFunction)THPTensor_stateless_(sinh), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"tan", (PyCFunction)THPTensor_stateless_(tan), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"atan", (PyCFunction)THPTensor_stateless_(atan), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"tanh", (PyCFunction)THPTensor_stateless_(tanh), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"sqrt", (PyCFunction)THPTensor_stateless_(sqrt), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"rsqrt", (PyCFunction)THPTensor_stateless_(rsqrt), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"ceil", (PyCFunction)THPTensor_stateless_(ceil), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"floor", (PyCFunction)THPTensor_stateless_(floor), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"round", (PyCFunction)THPTensor_stateless_(round), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"trunc", (PyCFunction)THPTensor_stateless_(trunc), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"frac", (PyCFunction)THPTensor_stateless_(frac), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"mean", (PyCFunction)THPTensor_stateless_(mean), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"var", (PyCFunction)THPTensor_stateless_(var), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"std", (PyCFunction)THPTensor_stateless_(std), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"norm", (PyCFunction)THPTensor_stateless_(norm), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"renorm", (PyCFunction)THPTensor_stateless_(renorm), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"dist", (PyCFunction)THPTensor_stateless_(dist), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"cinv", (PyCFunction)THPTensor_stateless_(cinv), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"neg", (PyCFunction)THPTensor_stateless_(neg), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"atan2", (PyCFunction)THPTensor_stateless_(atan2), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"pow", (PyCFunction)THPTensor_stateless_(pow), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE) || CUDA_FLOAT
  {"lerp", (PyCFunction)THPTensor_stateless_(lerp), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"linspace", (PyCFunction)THPTensor_stateless_(linspace), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"logspace", (PyCFunction)THPTensor_stateless_(logspace), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"histc", (PyCFunction)THPTensor_stateless_(histc), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"cmax", (PyCFunction)THPTensor_stateless_(cmax), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"cmin", (PyCFunction)THPTensor_stateless_(cmin), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"sum", (PyCFunction)THPTensor_stateless_(sum), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"prod", (PyCFunction)THPTensor_stateless_(prod), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"cumsum", (PyCFunction)THPTensor_stateless_(cumsum), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"cumprod", (PyCFunction)THPTensor_stateless_(cumprod), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"sign", (PyCFunction)THPTensor_stateless_(sign), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"trace", (PyCFunction)THPTensor_stateless_(trace), METH_VARARGS, NULL},
#endif
  {"add", (PyCFunction)THPTensor_stateless_(add), METH_VARARGS, NULL},
  {"sub", (PyCFunction)THPTensor_stateless_(sub), METH_VARARGS, NULL},
  {"mul", (PyCFunction)THPTensor_stateless_(mul), METH_VARARGS, NULL},
  {"div", (PyCFunction)THPTensor_stateless_(div), METH_VARARGS, NULL},
#if !IS_CUDA
  {"fmod", (PyCFunction)THPTensor_stateless_(fmod), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"remainder", (PyCFunction)THPTensor_stateless_(remainder), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"clamp", (PyCFunction)THPTensor_stateless_(clamp), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"dot", (PyCFunction)THPTensor_stateless_(dot), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"tril", (PyCFunction)THPTensor_stateless_(tril), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"triu", (PyCFunction)THPTensor_stateless_(triu), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"cross", (PyCFunction)THPTensor_stateless_(cross), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"eye", (PyCFunction)THPTensor_stateless_(eye), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"equal", (PyCFunction)THPTensor_stateless_(equal), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"diag", (PyCFunction)THPTensor_stateless_(diag), METH_VARARGS, NULL},
#endif
  {"lt", (PyCFunction)THPTensor_stateless_(lt), METH_VARARGS, NULL},
  {"gt", (PyCFunction)THPTensor_stateless_(gt), METH_VARARGS, NULL},
  {"le", (PyCFunction)THPTensor_stateless_(le), METH_VARARGS, NULL},
  {"ge", (PyCFunction)THPTensor_stateless_(ge), METH_VARARGS, NULL},
  {"eq", (PyCFunction)THPTensor_stateless_(eq), METH_VARARGS, NULL},
  {"ne", (PyCFunction)THPTensor_stateless_(ne), METH_VARARGS, NULL},
  {"min", (PyCFunction)THPTensor_stateless_(min), METH_VARARGS, NULL},
  {"max", (PyCFunction)THPTensor_stateless_(max), METH_VARARGS, NULL},
#if !IS_CUDA
  {"kthvalue", (PyCFunction)THPTensor_stateless_(kthvalue), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"mode", (PyCFunction)THPTensor_stateless_(mode), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"median", (PyCFunction)THPTensor_stateless_(median), METH_VARARGS, NULL},
#endif
  {"sort", (PyCFunction)THPTensor_stateless_(sort), METH_VARARGS, NULL},
#if CUDA_FLOAT || !IS_CUDA
  {"topk", (PyCFunction)THPTensor_stateless_(topk), METH_VARARGS, NULL},
#endif
  {"maskedSelect", (PyCFunction)THPTensor_stateless_(maskedSelect), METH_VARARGS, NULL},
  {"transpose", (PyCFunction)THPTensor_stateless_(transpose), METH_VARARGS, NULL},
  {"t", (PyCFunction)THPTensor_stateless_(t), METH_VARARGS, NULL},
  {"squeeze", (PyCFunction)THPTensor_stateless_(squeeze), METH_VARARGS, NULL},
#if !IS_CUDA
  {"nonzero", (PyCFunction)THPTensor_stateless_(nonzero), METH_VARARGS, NULL},
#endif
  {"indexSelect", (PyCFunction)THPTensor_stateless_(indexSelect), METH_VARARGS, NULL},
  {"indexCopy_", (PyCFunction)THPTensor_stateless_(indexCopy_), METH_VARARGS, NULL},
#if CUDA_FLOAT || !IS_CUDA
  {"indexAdd_", (PyCFunction)THPTensor_stateless_(indexAdd_), METH_VARARGS, NULL},
#endif
  {"indexFill_", (PyCFunction)THPTensor_stateless_(indexFill_), METH_VARARGS, NULL},
#if !IS_CUDA
  {"range", (PyCFunction)THPTensor_stateless_(range), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"gather", (PyCFunction)THPTensor_stateless_(gather), METH_VARARGS, NULL},
#endif
  {"addmm", (PyCFunction)THPTensor_stateless_(addmm), METH_VARARGS, NULL},
  {"addmv", (PyCFunction)THPTensor_stateless_(addmv), METH_VARARGS, NULL},
  {"addr", (PyCFunction)THPTensor_stateless_(addr), METH_VARARGS, NULL},
  {"ger", (PyCFunction)THPTensor_stateless_(ger), METH_VARARGS, NULL},
  {"mv", (PyCFunction)THPTensor_stateless_(mv), METH_VARARGS, NULL},
  {"mm", (PyCFunction)THPTensor_stateless_(mm), METH_VARARGS, NULL},
  {"bmm", (PyCFunction)THPTensor_stateless_(bmm), METH_VARARGS, NULL},
  {"addbmm", (PyCFunction)THPTensor_stateless_(addbmm), METH_VARARGS, NULL},
  {"baddbmm", (PyCFunction)THPTensor_stateless_(baddbmm), METH_VARARGS, NULL},
#if CUDA_FLOAT || !IS_CUDA
  {"addcmul", (PyCFunction)THPTensor_stateless_(addcmul), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"addcdiv", (PyCFunction)THPTensor_stateless_(addcdiv), METH_VARARGS, NULL},
#endif
#if !IS_CUDA
  {"randperm", (PyCFunction)THPTensor_stateless_(randperm), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"multinomial", (PyCFunction)THPTensor_stateless_(multinomial), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"rand", (PyCFunction)THPTensor_stateless_(rand), METH_VARARGS, NULL},
#endif
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  {"randn", (PyCFunction)THPTensor_stateless_(randn), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT
  {"rand", (PyCFunction)THPTensor_stateless_(rand), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT
  {"randn", (PyCFunction)THPTensor_stateless_(randn), METH_VARARGS, NULL},
#endif
#if CUDA_FLOAT || !IS_CUDA
  {"cat", (PyCFunction)THPTensor_stateless_(cat), METH_VARARGS, NULL},
#endif

  {NULL}
};
// PUT DEFINITIONS IN HERE PLEASE

#undef IS_CUDA
#undef CUDA_FLOAT
#undef THPIndexTensor
#undef THPIndexTensorClass
#undef THPBoolTensor
#undef THPBoolTensorClass
#undef RealStr
#undef AS_REAL
