#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorRandom.c"
#else

#if defined(TH_REAL_IS_BYTE)
void THTensor_(getRNGState)(THGenerator *_generator, THTensor *self)
{
  static const size_t size = sizeof(THGenerator);
  THGenerator *rng_state;
  THTensor_(resize1d)(self, size);
  THArgCheck(THTensor_(nElement)(self) == size, 1, "RNG state is wrong size");
  THArgCheck(THTensor_(isContiguous)(self), 1, "RNG state needs to be contiguous");
  rng_state = (THGenerator *)THTensor_(data)(self);
  THGenerator_copy(rng_state, _generator);
}

void THTensor_(setRNGState)(THGenerator *_generator, THTensor *self)
{
  static const size_t size = sizeof(THGenerator);
  THGenerator *rng_state;
  THArgCheck(THTensor_(nElement)(self) == size, 1, "RNG state is wrong size");
  THArgCheck(THTensor_(isContiguous)(self), 1, "RNG state needs to be contiguous");
  rng_state = (THGenerator *)THTensor_(data)(self);
  THArgCheck(THGenerator_isValid(rng_state), 1, "Invalid RNG state");
  THGenerator_copy(_generator, rng_state);
}

#endif

#endif
