#include <Python.h>

#include <stdbool.h>
#include <unordered_map>
#include <TH/TH.h>

#define WITH_NUMPY_IMPORT_ARRAY
#include "THP.h"

PyObject* module;
PyObject* tensor_classes;

PyObject *THPDoubleStorageClass = NULL;
PyObject *THPFloatStorageClass  = NULL;
PyObject *THPLongStorageClass   = NULL;
PyObject *THPIntStorageClass    = NULL;
PyObject *THPShortStorageClass  = NULL;
PyObject *THPCharStorageClass   = NULL;
PyObject *THPByteStorageClass   = NULL;

PyObject *THPDoubleTensorClass  = NULL;
PyObject *THPFloatTensorClass   = NULL;
PyObject *THPLongTensorClass    = NULL;
PyObject *THPIntTensorClass     = NULL;
PyObject *THPShortTensorClass   = NULL;
PyObject *THPCharTensorClass    = NULL;
PyObject *THPByteTensorClass    = NULL;

PyObject *THPDefaultTensorClass = NULL;
PyObject *THPGeneratorClass     = NULL;

// Used if no other generator is provided
THPGenerator *THPDefaultGenerator   = NULL;

static bool THPModule_loadClasses(PyObject *self)
{
#define ASSERT_NOT_NULL(ptr) if (!(ptr)) { THPUtils_setError("couldn't load classes"); return false; }
  PyObject *torch_module = PyImport_ImportModule("torch");
  if (!torch_module) {
    THPUtils_setError("class loader couldn't access torch module");
    return false;
  }
  PyObject* module_dict = PyModule_GetDict(torch_module);

  ASSERT_NOT_NULL(tensor_classes = PyMapping_GetItemString(module_dict, (char*)"_tensor_classes"));

  ASSERT_NOT_NULL(THPDoubleStorageClass = PyMapping_GetItemString(module_dict,(char*)"DoubleStorage"));
  ASSERT_NOT_NULL(THPFloatStorageClass  = PyMapping_GetItemString(module_dict,(char*)"FloatStorage"));
  ASSERT_NOT_NULL(THPLongStorageClass   = PyMapping_GetItemString(module_dict,(char*)"LongStorage"));
  ASSERT_NOT_NULL(THPIntStorageClass    = PyMapping_GetItemString(module_dict,(char*)"IntStorage"));
  ASSERT_NOT_NULL(THPShortStorageClass  = PyMapping_GetItemString(module_dict,(char*)"ShortStorage"));
  ASSERT_NOT_NULL(THPCharStorageClass   = PyMapping_GetItemString(module_dict,(char*)"CharStorage"));
  ASSERT_NOT_NULL(THPByteStorageClass   = PyMapping_GetItemString(module_dict,(char*)"ByteStorage"));

  ASSERT_NOT_NULL(THPDoubleTensorClass  = PyMapping_GetItemString(module_dict,(char*)"DoubleTensor"));
  ASSERT_NOT_NULL(THPFloatTensorClass   = PyMapping_GetItemString(module_dict,(char*)"FloatTensor"));
  ASSERT_NOT_NULL(THPLongTensorClass    = PyMapping_GetItemString(module_dict,(char*)"LongTensor"));
  ASSERT_NOT_NULL(THPIntTensorClass     = PyMapping_GetItemString(module_dict,(char*)"IntTensor"));
  ASSERT_NOT_NULL(THPShortTensorClass   = PyMapping_GetItemString(module_dict,(char*)"ShortTensor"));
  ASSERT_NOT_NULL(THPCharTensorClass    = PyMapping_GetItemString(module_dict,(char*)"CharTensor"));
  ASSERT_NOT_NULL(THPByteTensorClass    = PyMapping_GetItemString(module_dict,(char*)"ByteTensor"));

  THPDefaultTensorClass = THPDoubleTensorClass;

  return true;
#undef ASSERT_NOT_NULL
}


////////////////////////////////////////////////////////////////////////////////
// Copy handlers
////////////////////////////////////////////////////////////////////////////////

#include "ModuleCopy.h"

std::unordered_map<std::pair<PyObject *, PyObject *>, THPCopyFunction, pair_hasher> tensor_copy_handlers;
std::unordered_map<std::pair<PyObject *, PyObject *>, THPCopyFunction, pair_hasher> storage_copy_handlers;

#define COPY_METHODS(name) TH_CONCAT_2(name,_copy_handlers)
#define IMPLEMENT_COPY_WITH_WRAPPER(name)                                      \
bool TH_CONCAT_3(THPModule_,name,Copy)(PyObject *dst, PyObject *src)           \
{                                                                              \
  /* TODO: this won't work for subclasses, but is that a problem? */           \
  auto it = COPY_METHODS(name).find(std::make_pair((PyObject*)Py_TYPE(dst), (PyObject*)Py_TYPE(src))); \
  if (it == COPY_METHODS(name).end()) {                                        \
    THPUtils_setError("Copy function from %s to %s isn't implemented!", Py_TYPE(src)->tp_name, Py_TYPE(dst)->tp_name); \
    return false;                                                              \
  }                                                                            \
  (it->second)(dst, src);                                                      \
  return true;                                                                 \
}                                                                              \
                                                                               \
static PyObject * TH_CONCAT_3(THPModule_,name,CopyWrapper)(PyObject *unused, PyObject *args)\
{                                                                              \
  HANDLE_TH_ERRORS                                                             \
  /* TODO: check args */                                                       \
  PyObject *dst = PyTuple_GET_ITEM(args, 0);                                   \
  PyObject *src = PyTuple_GET_ITEM(args, 1);                                   \
  if (!TH_CONCAT_3(THPModule_,name,Copy)(dst, src)) {                          \
    return NULL;                                                               \
  }                                                                            \
  /* TODO: return dst? */                                                      \
  Py_RETURN_NONE;                                                              \
  END_HANDLE_TH_ERRORS                                                         \
}

IMPLEMENT_COPY_WITH_WRAPPER(tensor)
IMPLEMENT_COPY_WITH_WRAPPER(storage)
#undef COPY_METHODS
#undef IMPLEMENT_COPY_WITH_WRAPPER

#include "ModuleCopy.cpp"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static bool THPModule_assignStateless(PyObject *self)
{
#define INIT_STATELESS(type)                                                   \
  stateless = PyObject_Call((PyObject*)&TH_CONCAT_2(type, TensorStatelessType), arg, NULL); \
  if (!stateless) {                                                            \
    THPUtils_setError("stateless method initialization error");                \
    return false;                                                              \
  }                                                                            \
  if (PyObject_SetAttrString(TH_CONCAT_3(THP,type,TensorClass), STATELESS_ATTRIBUTE_NAME, stateless) == -1) { \
    THPUtils_setError("stateless method initialization error (on assignment)");\
  }
  PyObject *arg = PyTuple_New(0);
  PyObject *stateless;
  INIT_STATELESS(Double);
  INIT_STATELESS(Float);
  INIT_STATELESS(Long);
  INIT_STATELESS(Int);
  INIT_STATELESS(Short);
  INIT_STATELESS(Char);
  INIT_STATELESS(Byte);
  Py_DECREF(arg);
  return true;
#undef INIT_STATELESS
}

// Callback for python part. Used for additional initialization of python classes
static PyObject * THPModule_initExtension(PyObject *self)
{
  if (!THPModule_loadClasses(self))         return NULL;
  if (!THPModule_assignStateless(self))     return NULL;
  if (!THPModule_initCopy(self))            return NULL;

  return PyBool_FromLong(true);
}

static PyObject * THPModule_getNumThreads(PyObject *module)
{
#ifdef _OPENMP
  return PyLong_FromLong(omp_get_max_threads());
#else
  return PyLong_FromLong(1);
#endif
}

static PyObject * THPModule_setNumThreads(PyObject *module, PyObject *arg)
{
  if (!THPUtils_checkLong(arg))
    return NULL;
  // TODO: maybe throw an error to let people know it's a noop? or a warning?
#ifdef _OPENMP
  omp_set_num_threads(THPUtils_getLong(arg));
#endif
  return 0;
}

//
//static PyObject * THPModule_getRNGState(PyObject *module, PyObject *args)
//{
//  THGenerator *generator = THPDefaultGenerator->cdata;
//  if (args && PyTuple_Size(args) == 1 && THPGenerator_Check(PyTuple_GET_ITEM(args, 0))) {
//    generator = ((THPGenerator*)PyTuple_GET_ITEM(args, 0))->cdata;
//  } else if (args && PyTuple_Size(args) > 0) {
//    // TODO: better error message
//    THPUtils_setError("invalid arguments");
//    return NULL;
//  }
//  THByteTensorPtr _t = THByteTensor_new();
//  THByteTensor_getRNGState(generator, _t.get());
//  PyObject *_ret =  THPByteTensor_newObject(_t);
//  _t.release();
//  return _ret;
//}


static PyMethodDef TorchMethods[] = {

  {"_initExtension",  (PyCFunction)THPModule_initExtension,     METH_NOARGS,  NULL},
  {"_tensorCopy",     (PyCFunction)THPModule_tensorCopyWrapper, METH_VARARGS, NULL},
  {"_storageCopy",    (PyCFunction)THPModule_storageCopyWrapper, METH_VARARGS, NULL},
  {"getNumThreads",   (PyCFunction)THPModule_getNumThreads,     METH_NOARGS,  NULL},
  {NULL, NULL, 0, NULL}

};

#if PY_MAJOR_VERSION != 2
static struct PyModuleDef torchmodule = {
   PyModuleDef_HEAD_INIT,
   "torch._C",
   NULL,
   -1,
   TorchMethods
};
#endif

static void errorHandler(const char *msg, void *data)
{
  throw THException(msg);
}

static void errorHandlerArg(int argNumber, const char *msg, void *data)
{
  throw THArgException(msg, argNumber);
}

static void updateErrorHandlers()
{
  THSetErrorHandler(errorHandler, NULL);
  THSetArgErrorHandler(errorHandlerArg, NULL);
}

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init_C()
#else
PyMODINIT_FUNC PyInit__C()
#endif
{

#if PY_MAJOR_VERSION == 2
#define ASSERT_TRUE(cmd) if (!(cmd)) {PyErr_SetString(PyExc_ImportError, "initialization error"); return;}
#else
#define ASSERT_TRUE(cmd) if (!(cmd)) return NULL
#endif

#if PY_MAJOR_VERSION == 2
  ASSERT_TRUE(module = Py_InitModule("torch._C", TorchMethods));
#else
  ASSERT_TRUE(module = PyModule_Create(&torchmodule));
#endif
  ASSERT_TRUE(THPGenerator_init(module));

  ASSERT_TRUE(THPDoubleStorage_init(module));
  ASSERT_TRUE(THPFloatStorage_init(module));
  ASSERT_TRUE(THPLongStorage_init(module));
  ASSERT_TRUE(THPIntStorage_init(module));
  ASSERT_TRUE(THPShortStorage_init(module));
  ASSERT_TRUE(THPCharStorage_init(module));
  ASSERT_TRUE(THPByteStorage_init(module));

  ASSERT_TRUE(THPDoubleTensor_init(module));
  ASSERT_TRUE(THPFloatTensor_init(module));
  ASSERT_TRUE(THPLongTensor_init(module));
  ASSERT_TRUE(THPIntTensor_init(module));
  ASSERT_TRUE(THPShortTensor_init(module));
  ASSERT_TRUE(THPCharTensor_init(module));
  ASSERT_TRUE(THPByteTensor_init(module));

  THPDefaultGenerator = (THPGenerator*)THPGenerator_newObject();
  ASSERT_TRUE(THPDefaultGenerator != nullptr);

  updateErrorHandlers();

#if PY_MAJOR_VERSION == 2
#else
  return module;
#endif

#undef ASSERT_TRUE

}