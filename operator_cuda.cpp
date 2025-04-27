#include <torch/extension.h>

void square_cuda(at::Tensor x, at::Tensor y);

at::Tensor square(at::Tensor x) {
    auto y = torch::empty_like(x);
    square_cuda(x, y);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square", &square, "Elementwise square (CUDA)");
}
