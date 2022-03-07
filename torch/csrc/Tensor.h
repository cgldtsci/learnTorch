#ifndef THP_TENSOR_INC
#define THP_TENSOR_INC

#define THPTensor_(NAME)            TH_CONCAT_4(THP,Real,Tensor_,NAME)
#define THPTensor                   TH_CONCAT_3(THP,Real,Tensor)
#define THPTensorStr                TH_CONCAT_STRING_2(Real,Tensor)
#define THPTensorType               TH_CONCAT_3(THP,Real,TensorType)
#define THPTensorBaseStr            TH_CONCAT_STRING_2(Real,TensorBase)
#define THPTensorClass              TH_CONCAT_3(THP,Real,TensorClass)

#include "generic/Tensor.h"
#include <TH/THGenerateAllTypes.h>

#endif
