#ifndef  __UTILS__
#define __UTILS__
#include <functional>
#include <torch/torch.h>

namespace utils {    
    auto total_size(const torch::Tensor&) -> int64_t;
    auto explain(const torch::nn::Module&) -> void;
    auto noisy_sinus(const torch::Tensor&, double = 0.05) -> torch::Tensor;
    auto generate_xy(const uint32_t n_batch, 
                     const uint32_t n_dim,
                     std::function<torch::Tensor(const torch::Tensor&)> func
                     ) -> std::pair<torch::Tensor, torch::Tensor>;
    auto randint(uint32_t min_inc, uint32_t max_exc) -> uint32_t;
    
    /* 
     * Templates are defined & declared in header files.
     */
    template<typename T>
    auto average(const std::vector<T>& toavg) -> float {
        T sum = std::accumulate(toavg.cbegin(), toavg.cend(), T(0));
        return float(sum)/toavg.size();
    }

    template<typename T, typename U>
    auto sort_by(std::vector<T>& to_sort, 
                 const std::vector<U>& order)-> void {

        // this sort is not in place
        std::vector<std::pair<std::size_t, U>> indices_value;
        // create vector of pairs
        for(std::size_t i = 0; i < order.size(); ++i){
            indices_value.emplace_back(i, order[i]);
        }
        std::sort(indices_value.begin(), indices_value.end(), 
            []( const std::pair<std::size_t, U> &a, 
                const std::pair<std::size_t, U> &b) -> bool {
                return a.second < b.second; // 
        });
        std::vector<T> sorted;
        for(std::size_t i = 0; i < order.size(); ++i){
            sorted.emplace_back( std::move(to_sort[indices_value[i].first]) );
        }
        // to_sort is in weird state at this point - all its elements are moved-fron
        // move sorted into to_sort
       to_sort = std::move(sorted);
    }
}
#endif