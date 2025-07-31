#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void test_kernel(float *input, float *output, int size) {}

torch::Tensor test_cuda_function(torch::Tensor input) { return input + 1; }
