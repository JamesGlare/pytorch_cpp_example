#ifndef __EVOLVE__
#define __EVOLVE__
#include <torch/torch.h>
#include "models.h"
#include "config.h"
#include "utils.h"
namespace evolve {

    class Species {
        private:
            std::vector<uint32_t> state; // TODO make this a more general struct which is accepted by model ctors
            Species(std::vector<uint32_t>&& state); // private constructor
        public:
            Species(uint32_t max_size, uint32_t low, uint32_t high); // random constructor
            
            static auto breed(const Species& mother, const Species& father) -> Species;
            auto mutate(double p_depth_change, double std) -> void;
            inline auto get_state() const -> std::vector<uint32_t> { return state;};
    };
    
    auto evolve( float target_avg_loss,
                 uint32_t max_layers, 
                 uint32_t min_layer_size, 
                 uint32_t max_layer_size,
                 uint32_t max_rounds,
                 uint32_t population
                ) -> std::vector<models::mlp_ptr>;
}
#endif