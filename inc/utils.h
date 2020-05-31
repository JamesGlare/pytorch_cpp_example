#ifndef  __UTILS__
#define __UTILS__
#include <torch/torch.h>

namespace utils {    
    auto total_size(const torch::Tensor&) -> int64_t;
    auto explain(const torch::nn::Module&) -> void;
    auto func(const torch::Tensor&, double = 0.05) -> torch::Tensor;
    auto generate_xy(const uint32_t, 
                     const uint32_t
                     ) -> std::pair<torch::Tensor, torch::Tensor>;
}
#endif