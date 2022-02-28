#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/utils.cpp"
#else

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
