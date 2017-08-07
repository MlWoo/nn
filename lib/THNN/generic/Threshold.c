#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Threshold.c"
#else

#define TH_OMP_OVERHEAD_THRESHOLD 1000

void THNN_(Threshold_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal threshold_,
          accreal val_,
          bool inplace)
{
  real threshold = TH_CONVERT_ACCREAL_TO_REAL(threshold_);
  real val = TH_CONVERT_ACCREAL_TO_REAL(val_);
  
  ptrdiff_t inputSize = THTensor_(nElement)(input);
  int inputContig = THTensor_(isContiguous)(input)? 1:0;

  if (inplace)
  {
    if (inputContig) {
#ifdef _OPENMP
      int i = 0;
      real *inp = input->storage->data+input->storageOffset;
      #pragma omp parallel for if(inputSize > TH_OMP_OVERHEAD_THRESHOLD) 
      for (i = 0; i < inputSize; ++i) {
        real* input_data = inp + i;
        if(*input_data <= threshold)
            *input_data = val;
      }
#else
      TH_TENSOR_APPLY(real, input,
        if (*input_data <= threshold)
          *input_data = val;
      );
#endif      

    } else { 
      TH_TENSOR_APPLY(real, input,
        if (*input_data <= threshold)
          *input_data = val;
      );
    }
    THTensor_(set)(output, input);
  }
  else
  {
    THTensor_(resizeAs)(output, input);
#ifdef _OPENMP
    ptrdiff_t outputSize = THTensor_(nElement)(output);
    int outputContig = THTensor_(isContiguous)(output)? 1:0;
    TH_TENSOR_APPLY2_ADVANCED_INDEX2(outputSize, outputContig, inputContig, real, output, real, input,
      *output_data = (*input_data > threshold) ? *input_data : val;
    );
#else
    TH_TENSOR_APPLY2(real, output, real, input,
      *output_data = (*input_data > threshold) ? *input_data : val;
    );
#endif
  }
}

void THNN_(Threshold_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accreal threshold_,
          accreal val_,
          bool inplace)
{
  real threshold = TH_CONVERT_ACCREAL_TO_REAL(threshold_);
  real val = TH_CONVERT_ACCREAL_TO_REAL(val_);
  THNN_CHECK_NELEMENT(input, gradOutput);
  ptrdiff_t inputSize = THTensor_(nElement)(input);
  ptrdiff_t gradOutputSize = THTensor_(nElement)(gradOutput);
  int inputContig = THTensor_(isContiguous)(input)? 1:0;
  int gradOutputContig = THTensor_(isContiguous)(gradOutput)? 1:0;
  if (inplace)
  {
    TH_TENSOR_APPLY2_ADVANCED_INDEX2(gradOutputSize, gradOutputContig, inputContig, real, gradOutput, real, input,
      if ((*input_data) <= threshold)
        *gradOutput_data = 0;
    );
    THTensor_(set)(gradInput, gradOutput);
  }
  else
  {
    THTensor_(resizeAs)(gradInput, input);
    ptrdiff_t gradInputSize = THTensor_(nElement)(gradInput);
    int gradInputContig = THTensor_(isContiguous)(gradInput)? 1:0;
    TH_TENSOR_APPLY3_ADVANCED_INDEX2(gradInputSize, gradInputContig, gradOutputContig, inputContig, real, gradInput, real, gradOutput, real, input,
      if ((*input_data) > threshold)
        *gradInput_data = *gradOutput_data;
      else
        *gradInput_data = 0;
    );
  }
}

#endif
