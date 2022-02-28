#include <Python.h>
#include <stdarg.h>
#include <string>
#include "THP.h"

#include "generic/utils.cpp"
#include <TH/THGenerateAllTypes.h>

bool THPUtils_checkLong(PyObject *index) {
    return PyLong_Check(index) || PyInt_Check(index);
}

long THPUtils_unpackLong(PyObject *index) {
  if (PyLong_Check(index)) {
    return PyLong_AsLong(index);
  } else if (PyInt_Check(index)) {
    return PyInt_AsLong(index);
  } else {
    throw std::exception();
  }
}

int THPUtils_getLong(PyObject *index, long *result) {
  try {
    *result = THPUtils_unpackLong(index);
  } catch(...) {
    char err_string[512];
    snprintf (err_string, 512, "%s %s",
      "getLong expected int or long, but got type: ",
      index->ob_type->tp_name);
    PyErr_SetString(PyExc_RuntimeError, err_string);
    return 0;
  }
  return 1;
}


void THPUtils_setError(const char *format, ...)
{
  static const size_t ERROR_BUFFER_SIZE = 1000;
  char buffer[ERROR_BUFFER_SIZE];
  va_list fmt_args;

  va_start(fmt_args, format);
  vsnprintf(buffer, ERROR_BUFFER_SIZE, format, fmt_args);
  va_end(fmt_args);
  PyErr_SetString(PyExc_RuntimeError, buffer);
}

template<>
void THPPointer<PyObject>::free() {
  if (ptr)
    Py_DECREF(ptr);
}

template class THPPointer<PyObject>;

