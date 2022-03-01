#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/utils.cpp"
#else

bool THPUtils_(parseReal)(PyObject *value, real *result)
{
  try {
    *result = THPUtils_(unpackReal)(value);
  } catch (...) {
   char err_string[512];
   snprintf (err_string, 512, "%s %s",
       "parseReal expected long or float, but got type: ",
       value->ob_type->tp_name);
    PyErr_SetString(PyExc_RuntimeError, err_string);
    return false;
  }
  return true;
}


#ifdef THC_REAL_IS_HALF
#define CONVERT(expr) THC_float2half((expr))
#else
#define CONVERT(expr) (expr)
#endif

real THPUtils_(unpackReal)(PyObject *value)
{
  if (PyLong_Check(value)) {
    return (real)CONVERT(PyLong_AsLongLong(value));
  }  else if (PyInt_Check(value)) {
    return (real)CONVERT(PyInt_AsLong(value));
  } else if (PyFloat_Check(value)) {
    return (real)CONVERT(PyFloat_AsDouble(value));
  } else {
    throw std::exception();
  }
}

template<>
void THPPointer<THStorage>::free() {
  if (ptr)
    THStorage_(free)(LIBRARY_STATE ptr);
}

template<>
void THPPointer<THPStorage>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

template class THPPointer<THStorage>;
template class THPPointer<THPStorage>;

#endif
