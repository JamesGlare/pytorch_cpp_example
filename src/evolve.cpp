#include <functional>
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
    Species::Species(uint32_t max_size, uint32_t low, uint32_t high) {
        // 0. determine size
        auto size = utils::randint(1, max_size);
        // 1 Create vector with that size
        for(uint32_t i = 0; i < size; ++i)
            state.emplace_back(utils::randint(low, high));
    }

    auto Species::breed(const Species& mother, const Species& father) -> Species {
        const auto n_father = father.state.size();
        const auto n_mother = mother.state.size();

        auto shallow_parent = std::make_unique<Species>(father);
        auto deep_parent = std::make_unique<Species>(mother);

        if (n_father > n_mother) {
            shallow_parent.swap(deep_parent);
        } 
        // the child has the average size of its parents
        uint32_t n_child = (n_father + n_mother)/2;
        decltype(father.state) child_state;

        for(uint32_t i = 0; i < n_child; ++i) { 
            if (i < shallow_parent->state.size()){
                if (utils::randint(0,2)) {
                    child_state.emplace_back(shallow_parent->state[i]);
                } else {
                    child_state.emplace_back(deep_parent->state[i]);
                }
            } else{
                child_state.emplace_back(deep_parent->state[i]);
            }
        }
        return Species(std::move(child_state));
    }

    auto Species::mutate(double p_depth_change = 0.2, double std = 0.1) -> void {
        // add layer?
        uint32_t inv_p = 1.0/p_depth_change;
        if(utils::randint(0, inv_p) == 0) { // 1/3 chance of changing length
            if(utils::randint(0,2)==0){
              if(this->state.size() > 1)
                this->state.pop_back();
            } else {
                this->state.push_back(state.back()); // just duplicate value from back
            }
        }
        // determine variation
        for (auto& s : this->state){
            uint32_t min = std::max((1.-std)*this->state.size(),0.);
            uint32_t max = (1.+std)*this->state.size();
            s = utils::randint(min, max);
        }
    }

    auto evolution_step(const std::vector<Species>& state_pool) -> std::vector<Species> {
        // assumes score, reversed-order (highest score states first)
    }
    // ******************************************************************************************** 
    auto rewrite_model_pool(const std::vector<Species>& state_pool, 
                         std::vector<models::model_ptr>& model_pool,
                         std::function<models::model_ptr(const std::vector<uint32_t>&)> model_ctor
                         ) -> void {
        std::transform(state_pool.begin(), state_pool.end(), model_pool.begin(),
                      [model_ctor](const Species& s)-> models::model_ptr { 
                          return model_ctor(s.get_state());
                    }
        );
    }
    
    auto evolve( float target_avg_loss,
                 uint32_t max_layers, 
                 uint32_t min_layer_size, 
                 uint32_t max_layer_size,
                 uint32_t max_rounds,
                 uint32_t population,
                 std::function<models::model_ptr(const std::vector<uint32_t>&)> model_ctor = models::make_MLP
                 ) -> std::vector<models::model_ptr> {
        
        // 1. Initialize random propulation
        std::vector<Species> state_pool;
        for(uint32_t i = 0; i< population; ++i)
            state_pool.emplace_back(max_layers, min_layer_size, max_layer_size);

        // 2. fill model pool
        std::vector<models::model_ptr> model_pool;
        for(const Species& s : state_pool){
            model_pool.push_back(model_ctor(s.get_state()));
        }

        // 3. Do the actual evolution loop
        uint32_t round = 0;
        auto avg_loss = std::numeric_limits<float>::max();
        config::TrainConfig trainconfig{20, 100, 0.001};
        config::EvalConfig evalconfig{20, 60};
        std::vector<float> population_score(population,0);

        while(round < max_rounds 
              && avg_loss > target_avg_loss) {
            // sort model vector
            
            rewrite_model_pool(state_pool, model_pool, model_ctor);
            uint32_t i = 0;
            for(auto& model : model_pool) {
                // this is where we want to do threading
                train_model(model, trainconfig);
                population_score[i] = score_model(model, evalconfig).back();
                ++i;
            }
            model_pool = utils::sort_by(model_pool, population_score);
            avg_loss = utils::average(population_score);
            ++round;
        }
        
        // Return last model vector
        
        return model_pool;
    }
}