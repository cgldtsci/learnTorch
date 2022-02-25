#include <exception>
#include <stdexcept>
#include <string>

#define HANDLE_TH_ERRORS                                                       \
  try {

#define END_HANDLE_TH_ERRORS_RET(retval)                                       \
  } catch (std::exception &e) {                                                \
    PyErr_SetString(PyExc_RuntimeError, e.what());                             \
    return retval;                                                             \
  }

#define END_HANDLE_TH_ERRORS END_HANDLE_TH_ERRORS_RET(NULL)