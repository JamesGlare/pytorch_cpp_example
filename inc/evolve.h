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
            auto mutate() -> void;
    };

    
    auto evolve(float target_avg_loss,
                uint32_t max_layers, 
                uint32_t min_layer_size, 
                uint32_t max_layer_size,
                uint32_t max_rounds 
                ) -> std::vector<models::model_ptr>;
}
#endif