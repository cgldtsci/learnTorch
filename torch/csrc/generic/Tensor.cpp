#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.cpp"
#else

extern PyObject *THPTensorClass;

bool THPTensor_(IsSubclass)(PyObject *tensor)
{
  return PyObject_IsSubclass((PyObject*)Py_TYPE(tensor), (PyObject*)&THPTensorType);
}

static PyObject * THPTensor_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  PyObject *cdata_arg = NULL;                 // keyword-only arg - cdata pointer value
  THLongStorage *sizes_arg = NULL;            // a storage with sizes for a new tensor
  THTensor *tensor_arg = NULL;                // a tensor to be viewed on
  // TODO: constructor from storage
  PyObject *iterable_arg = NULL;              // an iterable, with new tensor contents
  std::vector<size_t> iterator_lengths;       // a queue storing lengths of iterables at each depth
  bool args_ok = true;

  if (kwargs && PyDict_Size(kwargs) == 1) {
    cdata_arg = PyDict_GetItemString(kwargs, "cdata");
    args_ok = cdata_arg != NULL;
  } else if (args && PyTuple_Size(args) == 1) {
    PyObject *arg = PyTuple_GET_ITEM(args, 0);
    if (THPTensor_(IsSubclass)(arg)) {
      tensor_arg = ((THPTensor*)arg)->cdata;
    } else if (THPLongStorage_IsSubclass(arg)) {
      sizes_arg = ((THPLongStorage*)arg)->cdata;
    } else if (THPUtils_checkLong(arg)) {
      sizes_arg = THPUtils_getLongStorage(args);
      args_ok = sizes_arg != nullptr;
    } else {
      iterable_arg = arg;
      Py_INCREF(arg);
      THPObjectPtr item = arg;
      THPObjectPtr iter;
      while ((iter = PyObject_GetIter(item)) != nullptr) {
        Py_ssize_t length = PyObject_Length(item);
        iterator_lengths.push_back(length);
        if (iterator_lengths.size() > 1000000) {
            THPUtils_setError("Counted more than 1,000,000 dimensions in a given iterable. "
                    "Most likely your items are also iterable, and there's no "
                    "way to infer how many dimensions should the tensor have.");
            return NULL;
        }
        // TODO length == 0 is an error too
        if (length == -1) {
          // TODO: error
          return NULL;
        }
        if (length > 0) {
          item = PyIter_Next(iter);
          if (item == nullptr) {
            // TODO: set error
            return NULL;
          }
        } else {
          break;
        }
      }
      if (iterator_lengths.size() > 1) {
        for (auto length: iterator_lengths) {
          if (length <= 0) {
            // TODO: error message
            THPUtils_setError("invalid size");
            return NULL;
          }
        }
      }
      args_ok = iterator_lengths.size() > 0;
      // We have accumulated some errors along the way.
      // Since we did all checking and ignored only the non-important
      // ones it's safe to clear them here.
      PyErr_Clear();
    }
    } else if (args && PyTuple_Size(args) > 0) {
    sizes_arg = THPUtils_getLongStorage(args);
    args_ok = sizes_arg != nullptr;
  }

  if (!args_ok) {
    // TODO: nice error mossage
    THPUtils_setError("invalid arguments");
    return NULL;
  }

  return PyLong_FromLong(1);

  END_HANDLE_TH_ERRORS

}
// TODO: implement equality
PyTypeObject THPTensorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C." THPTensorBaseStr,          /* tp_name */
  NULL,                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  NULL,       /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  NULL,           /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,   /* will be assigned in init */    /* tp_methods */
  0,   /* will be assigned in init */    /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPTensor_(pynew),                     /* tp_new */
};

bool THPTensor_(init)(PyObject *module)
{
//  THPTensorType.tp_methods = THPTensor_(methods);
//  THPTensorType.tp_members = THPTensor_(members);
  if (PyType_Ready(&THPTensorType) < 0)
    return false;
//  THPTensorStatelessType.tp_new = PyType_GenericNew;
//  if (PyType_Ready(&THPTensorStatelessType) < 0)
//    return false;

  PyModule_AddObject(module, THPTensorBaseStr, (PyObject *)&THPTensorType);
  return true;
}

#endif