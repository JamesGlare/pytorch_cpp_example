#include <functional>
#include <iostream>
#include "evolve.h"
#include <thread>

namespace evolve {
    using namespace torch::indexing;

    auto train_model(models::model_ptr model,
                    std::function<torch::Tensor(const torch::Tensor&)> func,
                    const config::TrainConfig& config
                    ) -> void {
        model->train();
        torch::optim::Adam optimizer(model->parameters(), config.lr);
        for(uint32_t batch_idx = 0; 
            batch_idx < config.n_sample/config.n_batch; 
            ++batch_idx){
            auto batch_xy = utils::generate_xy(config.n_batch, model->get_in_size(), func);
            optimizer.zero_grad();
            auto prediction = model->forward(batch_xy.first);
            auto loss = torch::mse_loss(prediction, batch_xy.second);
            loss.backward(); 
            optimizer.step();
        }
    }


    auto score_model(models::model_ptr model,
                     std::function<torch::Tensor(const torch::Tensor&)> func,
                     const config::TestConfig& config
                    ) -> std::vector<float> {
        model->eval();
        std::vector<float> losses;
        for (uint32_t batch_idx = 0; 
             batch_idx< config.n_sample/config.n_sample; 
             ++batch_idx) {
            auto batch_xy = utils::generate_xy(config.n_batch, model->get_in_size(), func);
            auto prediction = model->forward(batch_xy.first);
            auto loss = torch::mse_loss(prediction, batch_xy.second);
            losses.push_back(loss.item<float>());
        }
        return losses;
    }

    // ********************************************************************************************
    class Species {
        private:
            std::vector<uint32_t> state; // TODO make this a more general struct which is accepted by model ctors
            Species(std::vector<uint32_t>&& state); // private constructor
        public:
            Species(uint32_t max_size, uint32_t low, uint32_t high); // random constructor
            
            static auto breed(const Species& mother, const Species& father) -> Species;
            auto mutate(double p_depth_change, double std) -> void;
            auto fix_dimension(uint32_t n_dim_in, uint32_t n_dim_out) -> void;
            inline auto get_state() const -> std::vector<uint32_t> { return state;};
            inline auto size() const -> uint32_t {return state.size(); };
            friend auto operator<<(std::ostream& os, const Species& s) -> std::ostream&;
    };
    
    Species::Species(std::vector<uint32_t>&& _state) : state(_state) {}
    
    Species::Species(uint32_t max_size, uint32_t low, uint32_t high) {
        // 0. determine size
        auto size = utils::randint(2, max_size);
        // 1 Create vector with that size
        for(uint32_t i = 0; i < size; ++i)
            state.emplace_back(utils::randint(low, high));
    }

    auto Species::fix_dimension(uint32_t n_dim_in, uint32_t n_dim_out) -> void {
        if (!state.empty()){
            state.front() = n_dim_in;
            state.back() = n_dim_out;
        }
    }

    auto operator<<(std::ostream& os, const Species& s) -> std::ostream& {
        os << s.state.front() << " " << s.state.back() << '\n';
        return os;
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

    auto Species::mutate(double p_depth_change, double std) -> void {
        // add layer?
        if (p_depth_change > 0.){
            uint32_t inv_p = 1.0/p_depth_change;
            if(utils::randint(0, inv_p) == 0) { // 1/3 chance of changing length
                if(utils::randint(0,2)==0){
                if(this->state.size() > 2)
                    this->state.pop_back();
                } else {
                    this->state.push_back(state.back()); // just duplicate value from back
                }
            }
        }
        if ( std > 0.){
            // determine variation
            for (auto& s : this->state){
                uint32_t min = std::max((1.-std)*this->state.size(),0.);
                uint32_t max = (1.+std)*this->state.size();
                s = utils::randint(min, max);
            }
        }
    }

    auto evolution_step(std::vector<Species>& state_pool,
                        const config::EvolConfig& evol_config
                        ) -> void {
        // assumes score, reversed-order (highest score states first)
        uint32_t middle = state_pool.size()/2;
        for(uint32_t i = 1; i < middle; ++i) {
            // just redraw the bottom layers except the last
            // by breeding the queen of the hill
            // with everyone on the hill (=top half)
            state_pool[middle - 1 + i] = Species::breed(  state_pool[0],  state_pool[i] );
        }
        // mutate everyone except the last one which gets redrawn anyway
        std::for_each(state_pool.begin(), state_pool.end()-1, 
            [&evol_config](Species& s) -> void { 
                    s.mutate(evol_config.pchange, evol_config.std);
            });
        for(uint32_t i = 0; i < state_pool.size()-1; ++i){
            double T = i/double(state_pool.size()-1);
            state_pool[i].mutate(T*evol_config.pchange, T*evol_config.std);
        }
        // the worst state is just redrawn
        state_pool.back() = Species(evol_config.max_layers, 
                                    evol_config.min_layer_size, 
                                    evol_config.max_layer_size);
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
    
    auto evolve(  float target_avg_loss,
                  const config::EvolConfig& evol_config,
                  const config::TrainConfig& train_config,
                  const config::TestConfig& test_config,
                  uint32_t n_dim_in,
                  uint32_t n_dim_out,
                  std::function<torch::Tensor(const torch::Tensor&)> target,
                  std::function<models::model_ptr(const std::vector<uint32_t>&)> model_ctor
                 ) -> std::vector<models::model_ptr> {
        // 1. Initialize random propulation
        std::vector<Species> state_pool;
        for(uint32_t i = 0; i < evol_config.population; ++i)
            state_pool.emplace_back(evol_config.max_layers, evol_config.min_layer_size, evol_config.max_layer_size);
        // pin input and output dimensions
        std::for_each(state_pool.begin(), state_pool.end(), 
                        [n_dim_in, n_dim_out](Species& s) { s.fix_dimension(n_dim_in, n_dim_out); });
        
        // 2. fill model pool
        std::vector<models::model_ptr> model_pool;
        for(const Species& s : state_pool){
            model_pool.push_back(model_ctor(s.get_state()));
        }
        // 3. Do the actual evolution loop
        uint32_t round = 0;
        auto avg_loss = std::numeric_limits<float>::max();
        
        std::vector<float> population_score(evol_config.population,0);
        while(round < evol_config.max_rounds 
              && avg_loss > target_avg_loss) {
            // sort model vector        
            rewrite_model_pool(state_pool, model_pool, model_ctor);
            uint32_t i = 0;
            for(auto& model : model_pool) {
                // this is where we want to do threading
                train_model(model, target, train_config);
                population_score[i] = score_model(model, target, test_config).back();
                ++i;
            }
            utils::sort_by(state_pool, population_score);

            // evolve
            evolution_step(state_pool, evol_config);
            // pin dimensions
            std::for_each(state_pool.begin(), state_pool.end(), 
                        [n_dim_in, n_dim_out](Species& s) { s.fix_dimension(n_dim_in, n_dim_out); });
            avg_loss = utils::average(population_score);
            //print results to std::out
            std::cout << round << " " << avg_loss << " " 
                      << *std::min_element(population_score.begin(), population_score.end()) <<'\n';
            ++round;
        }
        // Return last model vector
        return model_pool;
    }
}