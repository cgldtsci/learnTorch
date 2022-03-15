#ifndef THP_MODULE_INC
#define THP_MODULE_INC


#define STATELESS_ATTRIBUTE_NAME "_torch"

extern PyObject *THPDoubleStorageClass;
extern PyObject *THPFloatStorageClass;
extern PyObject *THPLongStorageClass;
extern PyObject *THPIntStorageClass;
extern PyObject *THPShortStorageClass;
extern PyObject *THPCharStorageClass;
extern PyObject *THPByteStorageClass;

extern PyObject *THPDoubleTensorClass;
extern PyObject *THPFloatTensorClass;
extern PyObject *THPLongTensorClass;
extern PyObject *THPIntTensorClass;
extern PyObject *THPShortTensorClass;
extern PyObject *THPCharTensorClass;
extern PyObject *THPByteTensorClass;

#endif