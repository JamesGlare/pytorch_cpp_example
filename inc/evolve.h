#ifndef __EVOLVE__
#define __EVOLVE__
#include <torch/torch.h>
#include "models.h"
#include "config.h"
#include "utils.h"
namespace evolve {
     auto evolve( float target_avg_loss,
                  const config::EvolConfig& evolconfig,
                  const config::TrainConfig& trainconfig,
                  const config::TestConfig& test_config,
                  uint32_t n_dim_in,
                  uint32_t n_dim_out,
                  std::function<torch::Tensor(const torch::Tensor&)> target,
                  std::function<models::model_ptr(const std::vector<uint32_t>&)> model_ctor
                  ) -> std::vector<models::model_ptr>;
}
#endif