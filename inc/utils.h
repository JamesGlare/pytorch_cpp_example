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
    
    /* 
     * Templates are defined & declared in header files.
     */
    template<typename T>
    auto average(const std::vector<T>& toavg) -> float {
        T sum = std::accumulate(toavg.begin(), toavg.end(), T(0));
        return float(sum)/toavg.size();
    }

    template<typename T, typename U>
    auto sort_by(const std::vector<T>& to_sort, 
                 const std::vector<U>& order) -> std::vector<T> {
        // this sort is not in place
        std::vector<uint32_t> indices(to_sort.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                [&](int A, int B) -> bool {
                    return order[A] < order[B];
                }
            );
        std::vector<T> sorted;
        std::transform(indices.begin(), indices.end(), std::back_inserter(sorted),
            [to_sort](uint32_t ii) -> T {
                return to_sort[ii];
            }
        );
        return sorted;
    }
}
#endif