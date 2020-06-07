#include "evolve.h"
namespace evolve {
    using namespace torch::indexing;

    auto train_model(models::torch_ptr model,
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


    auto score_model(models::torch_ptr model,
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
}