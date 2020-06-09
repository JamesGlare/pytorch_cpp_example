#ifndef  __UTILS__
#define __UTILS__
#include <torch/torch.h>

namespace utils {    
    auto total_size(const torch::Tensor&) -> int64_t;
    auto explain(const torch::nn::Module&) -> void;
    auto func(const torch::Tensor&, double = 0.05) -> torch::Tensor;
    auto generate_xy(const uint32_t, 
                     const uint32_t // TODO make this accept a f(torch::tensor) -> torch::tensor
                     ) -> std::pair<torch::Tensor, torch::Tensor>;
    auto randint(uint32_t min_inc, uint32_t max_exc) -> uint32_t;
}
#endif