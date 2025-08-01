#include <torch/extension.h>

torch::Tensor test_function(torch::Tensor input) { return input; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test_function", &test_function, "Test function");
}
