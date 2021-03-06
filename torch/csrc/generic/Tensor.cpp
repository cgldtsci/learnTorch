#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.cpp"
#else

extern PyObject *THPTensorClass;

extern PyObject *THPTensorClass;

#include "TensorMethods.cpp"

PyObject * THPTensor_(newObject)(THTensor *ptr)
{
  // TODO: error checking
  PyObject *args = PyTuple_New(0);
  PyObject *kwargs = Py_BuildValue("{s:K}", "cdata", (unsigned long long) ptr);
  PyObject *instance = PyObject_Call(THPTensorClass, args, kwargs);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  return instance;
}

bool THPTensor_(IsSubclass)(PyObject *tensor)
{
  return PyObject_IsSubclass((PyObject*)Py_TYPE(tensor), (PyObject*)&THPTensorType);
}

static void THPTensor_(dealloc)(THPTensor* self)
{
  THTensor_(free)(LIBRARY_STATE self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
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

  THPTensorPtr self = (THPTensor *)type->tp_alloc(type, 0);
  if (self != nullptr) {
    if (cdata_arg) {
      self->cdata = (THTensor*)PyLong_AsVoidPtr(cdata_arg);
    } else if (sizes_arg) {
      self->cdata = THTensor_(newWithSize)(LIBRARY_STATE sizes_arg, nullptr);
    } else if (tensor_arg) {
      self->cdata = THTensor_(newWithTensor)(LIBRARY_STATE tensor_arg);
    } else if (iterable_arg && iterator_lengths.size() == 1 && iterator_lengths[0] == 0) {
      self->cdata = THTensor_(new)(LIBRARY_STATE_NOARGS);
    } else if (iterable_arg) {
      size_t iter_depth = iterator_lengths.size();
      std::stack<THPObjectPtr> iterator_stack;
      std::vector<size_t> items_processed(iter_depth);
      Py_INCREF(iterable_arg);
      THPObjectPtr item = iterable_arg;
      PyObject *iter;
      while (iterator_stack.size() != iter_depth) {
        iter = PyObject_GetIter(item);
        if (!iter) {
          THPUtils_setError("inconsistent iterator depth");
          return NULL;
        }
        iterator_stack.emplace(iter);
        item = PyIter_Next(iter);
        if (item == nullptr) {
          THPUtils_setError("error or empty iter");
          return NULL;
        }
      }
      THLongStoragePtr sizes = THLongStorage_newWithSize(iter_depth);
      long *sizes_data = sizes->data;
      for (size_t s: iterator_lengths) {
        *sizes_data++ = s;
      }
      THTensorPtr tensor = THTensor_(newWithSize)(LIBRARY_STATE sizes, NULL);

      // TODO CUDA
#ifndef THC_GENERIC_FILE
#define SET_ITEM if (!THPUtils_(parseReal)(item, data++)) return NULL
      real *data = tensor->storage->data;

#endif
      SET_ITEM;
      items_processed[iter_depth-1]++;

      while (!iterator_stack.empty()) {
        PyObject *iter = iterator_stack.top().get();
        // Parse items
        if (iterator_stack.size() == iter_depth) {
          while ((item = PyIter_Next(iter))) {
            SET_ITEM;
            items_processed[iter_depth-1]++;
          }
          if (items_processed[iter_depth-1] != iterator_lengths[iter_depth-1]) {
            THPUtils_setError("inconsistent size");
            return NULL;
          }
          iterator_stack.pop(); // this deallocates the iter
        // Iterate on lower depths
        } else {
          item = PyIter_Next(iter);
          if (item == nullptr) {
            if (PyErr_Occurred())
              return NULL;
            if (items_processed[iterator_stack.size()-1]) {
              THPUtils_setError("inconsistent size");
              return NULL;
            }
            iterator_stack.pop(); // this deallocates the iter
          } else {
            PyObject *new_iter = PyObject_GetIter(item);
            if (!new_iter) {
              THPUtils_setError("non-iterable item");
              return NULL;
            }
            items_processed[iterator_stack.size()] = 0;
            iterator_stack.emplace(new_iter);
          }
        }
      }
      self->cdata = tensor.release();
    } else {
      self->cdata = THTensor_(new)(LIBRARY_STATE_NOARGS);
    }

    if (self->cdata == NULL)
      return NULL;
  }

  return (PyObject *)self.release();

  END_HANDLE_TH_ERRORS
}

// TODO: implement equality
PyTypeObject THPTensorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C." THPTensorBaseStr,          /* tp_name */
  sizeof(THPTensor),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPTensor_(dealloc),       /* tp_dealloc */
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

static struct PyMemberDef THPTensor_(members)[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THPTensor, cdata), READONLY, NULL},
  {NULL}
};

typedef struct {
  PyObject_HEAD
} THPTensorStateless;


PyTypeObject THPTensorStatelessType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C." THPTensorBaseStr ".stateless", /* tp_name */
  sizeof(THPTensorStateless),            /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved / tp_compare */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
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
  THPTensor_stateless_(methods),         /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  0,                                     /* tp_new */
  0,                                     /* tp_free */
  0,                                     /* tp_is_gc */
  0,                                     /* tp_bases */
  0,                                     /* tp_mro */
  0,                                     /* tp_cache */
  0,                                     /* tp_subclasses */
  0,                                     /* tp_weaklist */
};

bool THPTensor_(init)(PyObject *module)
{
  THPTensorType.tp_methods = THPTensor_(methods);
  THPTensorType.tp_members = THPTensor_(members);
  if (PyType_Ready(&THPTensorType) < 0)
    return false;
  THPTensorStatelessType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&THPTensorStatelessType) < 0)
    return false;

  PyModule_AddObject(module, THPTensorBaseStr, (PyObject *)&THPTensorType);
  return true;
}

#endif