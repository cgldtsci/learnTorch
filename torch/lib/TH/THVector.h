#ifndef TH_VECTOR_INC
#define TH_VECTOR_INC

#include "THGeneral.h"

#define THVector_(NAME) TH_CONCAT_4(TH,Real,Vector_,NAME)


/* If SSE2 not defined, then generate plain C operators */
#include "generic/THVector.c"
#include "THGenerateFloatTypes.h"

/* For non-float types, generate plain C operators */
#include "generic/THVector.c"
#include "THGenerateIntTypes.h"


#endif
