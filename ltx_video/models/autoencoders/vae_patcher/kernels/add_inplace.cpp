#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/python.h>

#include <vector>

void inplace_add_cuda(
    at::Tensor& x,
    at::Tensor& input2,
    int offset,
    cudaStream_t stream
);



at::Tensor inplace_add(at::Tensor &x, at::Tensor &workspace, int64_t offset) {
    at::cuda::CUDAGuard device_guard{x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    inplace_add_cuda(
        x,
        workspace,
        offset,
        stream
    );
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("inplace_add", &inplace_add, "inplace_add wrapper");
}

TORCH_LIBRARY(inplace_add, m) {
    m.def("inplace_add(Tensor x, Tensor input2, int offset) -> Tensor");
    m.impl("inplace_add", inplace_add);
}
