#include "evolve.h"

namespace evolve {
    using namespace torch::indexing;

    auto train_model(models::model_ptr model,
                    const config::TrainConfig& config
                    ) -> void {
        model->train();
        torch::optim::Adam optimizer(model->parameters(), config.lr);
        for(uint32_t batch_idx = 0; 
            batch_idx < config.n_sample/config.n_batch; 
            ++batch_idx){
            auto batch_xy = utils::generate_xy(config.n_batch, model->get_in_size());
            optimizer.zero_grad();
            auto prediction = model->forward(batch_xy.first);
            auto loss = torch::mse_loss(prediction, batch_xy.second);
            loss.backward(); 
            optimizer.step();
        }
    }


    auto score_model(models::model_ptr model,
                     const config::EvalConfig& config
                    ) -> std::vector<float> {
        model->eval();
        std::vector<float> losses;
        for (uint32_t batch_idx = 0; 
             batch_idx< config.n_sample/config.n_sample; 
             ++batch_idx) {
            auto batch_xy = utils::generate_xy(config.n_batch, model->get_in_size());
            auto prediction = model->forward(batch_xy.first);
            auto loss = torch::mse_loss(prediction, batch_xy.second);
            losses.push_back(loss.item<float>());
        }
        return losses;
    }

    // ********************************************************************************************
    Species::Species(std::vector<uint32_t>&& _state) : state(_state) {}
    auto Species::init_random(uint32_t max_size, uint32_t low, uint32_t high) -> Species {
        // 0. determine size
        auto size = utils::randint(1, max_size);
        // 1 Crate vector with that size
        std::vector<uint32_t> state(size);
        // 2 fill with randint values
        std::generate(state.begin(), state.end(), [low, high](){return utils::randint(low, high);});
        // 3 create species instance
        return Species(std::move(state));
    }
    auto Species::breed(const Species& mother, const Species& father) -> Species {
        const auto n_father = father.state.size();
        const auto n_mother = mother.state.size();
        if (n_father < n_mother) {
            auto shallow_parent = father;
            auto deep_parent = mother;
        } else {
            auto shallow_parent = mother;
            auto deep_parent = father;
        }
        uint32_t n_child = (n_father + n_mother)/2;
        std::vector<uin32_t> child_state(n_child);
        std::generate(child_state.begin(), child_state.end(),
            [father.state, mother.state](){ 
                                        if(utils::randint(0,2)){
                                            
                                        }
             });
    }
    // ******************************************************************************************** 

    auto evolve( float target_avg_loss,
                 uint32_t max_layers, 
                 uint32_t min_layer_size, 
                 uint32_t max_layer_size,
                 uint32_t max_rounds 
                 ) -> std::vector<models::model_ptr> {
        // 1. Initialize random propulation

    }
}