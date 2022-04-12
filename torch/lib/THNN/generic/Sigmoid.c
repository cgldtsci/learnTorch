#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Sigmoid.c"
#else

void THNN_(Sigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THTensor_(resizeAs)(output, input);

  TH_TENSOR_APPLY2(real, output, real, input,
    *output_data = 1./(1.+ exp(- *input_data));
  );
}

#endif
