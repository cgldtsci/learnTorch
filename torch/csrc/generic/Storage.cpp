#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Storage.cpp"
#else

/* A pointer to RealStorage class defined later in Python */
extern PyObject *THPStorageClass;

PyObject * THPStorage_(newObject)(THStorage *ptr)
{
  // TODO: error checking
  PyObject *args = PyTuple_New(0);
//  PyObject *kwargs = Py_BuildValue("{s:N}", "cdata", PyLong_FromVoidPtr(ptr));
//  PyObject *instance = PyObject_Call(THPStorageClass, args, kwargs);
  Py_DECREF(args);
//  Py_DECREF(kwargs);
//  return instance;
    return args;
}

#endif