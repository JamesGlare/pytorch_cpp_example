#ifndef __EVOLVE__
#define __EVOLVE__
#include <torch/torch.h>
#include <functional>
#include "models.h"
#include "config.h"
#include "utils.h"
namespace evolve {
    class Species {
        private:
            std::vector<uint32_t> state;
            Species(std::vector<uint32_t>&& state); // constructor
        public:
            static auto breed(const Species& mother, const Species& father) -> Species;
            static auto init_random(uint32_t max_size, uint32_t low, uint32_t high) -> Species;
            auto mutate(double p_depth_change, double std) -> void;
            inline auto get_state() const -> const std::vector<uint32_t>& { return state;};
    };

    
    auto evolve( float target_avg_loss,
                 uint32_t max_layers, 
                 uint32_t min_layer_size, 
                 uint32_t max_layer_size,
                 uint32_t max_rounds,
                 uint32_t population
                ) -> std::vector<models::model_ptr>;
}
#endif