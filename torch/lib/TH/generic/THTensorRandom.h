#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorRandom.h"
#else


#if defined(TH_REAL_IS_BYTE)
TH_API void THTensor_(getRNGState)(THGenerator *_generator, THTensor *self);
TH_API void THTensor_(setRNGState)(THGenerator *_generator, THTensor *self);
#endif

#endif
