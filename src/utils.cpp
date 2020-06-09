#include <math.h>
#include <stdlib.h>
#include "utils.h"

namespace utils {
    /*
     * Returns total number of parameters.
     */
    auto total_size(const torch::Tensor& t) -> int64_t {
        int64_t result = 1;
        for(int64_t s = 0; s < t.dim(); ++s){
            result*= t.size(s);
        } 
        return result;
    }
    /*
     * Print sizes of layers of torch net.
     */
    auto explain(const torch::nn::Module& net) -> void {
        int64_t summed_sizes = 0, temp = 0;
        for(const auto& p : net.named_parameters()){
            temp = total_size(p.value());
            std::cout<< p.key() << " contains "<< temp << " ...\n";
            summed_sizes += temp;
        }
        std::cout << "\n\t=> TOTAL "<< summed_sizes << " parameters." << std::endl;
    }
    auto generate_xy(const uint32_t n_batch, 
                     const uint32_t n_dim
                     ) -> std::pair<torch::Tensor, torch::Tensor> {
        auto x = torch::rand({n_batch, n_dim});
        return {x,func(x)};
    }
    /* 
     * Target function, which we try to learn here.
     */
    auto func(const torch::Tensor& x, double std) -> torch::Tensor {
        auto noise = std*torch::rand_like(x);
        return torch::sin(M_PI*x) + noise;
    }

    auto randint(uint32_t min_inc, uint32_t max_exc) -> uint32_t {
        // make sure you've set the seed using srand(int seed)
        return min_inc + rand() % max_exc;
    }
}