#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/utils.cpp"
#else


// TODO This doesn't need to be in generic file
bool THPUtils_(parseSlice)(PyObject *slice, Py_ssize_t len, Py_ssize_t *ostart, Py_ssize_t *ostop, Py_ssize_t *oslicelength)
{
  Py_ssize_t start, stop, step, slicelength;
  if (PySlice_GetIndicesEx(
// https://bugsfiles.kde.org/attachment.cgi?id=61186
#if PY_VERSION_HEX >= 0x03020000
         slice,
#else
         (PySliceObject *)slice,
#endif
         len, &start, &stop, &step, &slicelength) < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Got an invalid slice");
    return false;
  }
  if (step != 1) {
    PyErr_SetString(PyExc_RuntimeError, "Only step == 1 supported");
    return false;
  }
  *ostart = start;
  *ostop = stop;
  if(oslicelength)
    *oslicelength = slicelength;
  return true;
}
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
