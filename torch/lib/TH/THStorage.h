#ifndef TH_STORAGE_INC
#define TH_STORAGE_INC

#include "THGeneral.h"
#include "THAllocator.h"

#define THStorage        TH_CONCAT_3(TH,Real,Storage)
#define THStorage_(NAME) TH_CONCAT_4(TH,Real,Storage_,NAME)

#include "generic/THStorage.h"
#include "THGenerateAllTypes.h"

#endif
