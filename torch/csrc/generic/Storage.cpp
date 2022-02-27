#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Storage.cpp"
#else

/* A pointer to RealStorage class defined later in Python */
extern PyObject *THPStorageClass;
//
//PyObject * THPStorage_(newObject)(THStorage *ptr)
//{
//  // TODO: error checking
//  PyObject *args = PyTuple_New(0);
////  PyObject *kwargs = Py_BuildValue("{s:N}", "cdata", PyLong_FromVoidPtr(ptr));
////  PyObject *instance = PyObject_Call(THPStorageClass, args, kwargs);
//  Py_DECREF(args);
////  Py_DECREF(kwargs);
////  return instance;
//    return args;
//}

static PyObject * THPStorage_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS

  THPStoragePtr self = (THPStorage *)type->tp_alloc(type, 0);
  return type->tp_alloc(type,0);
//  return (PyObject *)self;
//
//  if (self == nullptr) {
//    THPUtils_setError("test nullptr error");
//
//  }
  THPUtils_setError("test pynew self error");

  return NULL;

//  return (PyObject *)self.release();

  END_HANDLE_TH_ERRORS

}


// TODO: implement equality
PyTypeObject THPStorageType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C." THPStorageBaseStr,         /* tp_name */
  NULL,                    /* tp_basicsize */
  0,                                     /* tp_itemsize */
  NULL,      /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  NULL,          /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  NULL,   /* will be assigned in init */    /* tp_methods */
  NULL,   /* will be assigned in init */    /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPStorage_(pynew),                    /* tp_new */
};

bool THPStorage_(init)(PyObject *module)
{
//  THPStorageType.tp_methods = THPStorage_(methods);
//  THPStorageType.tp_members = THPStorage_(members);
  if (PyType_Ready(&THPStorageType) < 0)
    return false;
  Py_INCREF(&THPStorageType);
  PyModule_AddObject(module, THPStorageBaseStr, (PyObject *)&THPStorageType);
  return true;
}
#endif