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
        int64_t summed_sizes = 0;
        for(const auto& p : net.named_parameters()){            
            summed_sizes += total_size(p.value());
            if (p.key().find("bias") == std::string::npos)
                std::cout  << total_size(p.value()) << " | ";
        }
        auto depth = net.parameters().size()/2;
        std::cout << "Depth: " << depth << " | Total: "<< summed_sizes << std::endl;
    }
    auto generate_xy(const uint32_t n_batch, 
                     const uint32_t n_dim,
                     std::function<torch::Tensor(const torch::Tensor&)> func
                     ) -> std::pair<torch::Tensor, torch::Tensor> {
        auto x = torch::rand({n_batch, n_dim});
        return {x,func(x)};
    }
    /* 
     * Target function, which we try to learn here.
     */
    auto noisy_sinus(const torch::Tensor& x, double std) -> torch::Tensor {
        auto noise = std*torch::rand_like(x);
        return torch::sin(M_PI*x) + noise;
    }

    auto randint(uint32_t min_inc, uint32_t max_exc) -> uint32_t {
        if (min_inc >= max_exc){
            return min_inc;
        }
        // make sure you've set the seed using srand(int seed)
        return min_inc + rand() % max_exc;
    }
}