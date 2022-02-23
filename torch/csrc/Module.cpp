#include <Python.h>

#include <TH/TH.h>

#include "THP.h"

PyObject* module;

static bool THPModule_loadClasses(PyObject *self)
{
#define ASSERT_NOT_NULL(ptr) if (!(ptr)) { THPUtils_setError("couldn't load classes"); return false; }
  PyObject *torch_module = PyImport_ImportModule("torch");
  if (!torch_module) {
    THPUtils_setError("class loader couldn't access torch module");
    return false;
  }

  return true;
#undef ASSERT_NOT_NULL

}

// Callback for python part. Used for additional initialization of python classes
static PyObject * THPModule_initExtension(PyObject *self)
{
  if (!THPModule_loadClasses(self))         return NULL;

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

static PyMethodDef TorchMethods[] = {

  {"_initExtension",  (PyCFunction)THPModule_initExtension,     METH_NOARGS,  NULL},
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

#if PY_MAJOR_VERSION == 2
#else
  return module;
#endif

#undef ASSERT_TRUE

}