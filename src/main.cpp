#include <iostream>
#include <thread>
#include <torch/torch.h>
#include "models.h"
#include "config.h"
#include "evolve.h"
/* Three things to set up here. 
 * 1. Cmake (quick start + CMakeLists.txt)
 * 2. Debugger (gdb, launch.json)
 * 3. Intellisense (c_cpp_properties -> two include paths for pytorch)
 */ 

// If you *want* to see assembly
// just go make src/main.s
// Create your own tasks in vscode
// https://code.visualstudio.com/Docs/editor/tasks

// For debugging, use the gbp + vscode
// need to have compiled in debug mode
// make sure you've checked up launch.json

auto main(int, char**) -> int {
    torch::manual_seed(42);

    const float target_loss = 0.05;
    const double noise_std = 0.05;
    const uint32_t n_dim_in = 5;
    const uint32_t n_dim_out = 5;
    constexpr config::EvolConfig evolution_config{
                    /* max layer */  5,
                    /* min width */  2,
                    /* max width */  50,
                    /* max rounds */ 30,
                    /* population */ 30,
                    /* depth mut prob */ 0.3,
                    /* width change std */ 0.1
                    };
    constexpr config::TrainConfig train_config{
                    /* batch */  20, 
                    /* sample */ 2000, 
                    /* lr */     0.001
                };
    constexpr config::TestConfig test_config{
                    /* batch */  20, 
                    /* sample */ 1000
    };

    auto models = evolve::evolve(target_loss, evolution_config, train_config, test_config,
                                 n_dim_in, n_dim_out,
                                 [&noise_std](const torch::Tensor& t) 
                                 { return utils::noisy_sinus(t, noise_std); },
                                 models::make_MLP
                                );
    std::for_each(models.begin(), models.end(), 
                 [](models::model_ptr& model){ utils::explain(*model); });
    return EXIT_SUCCESS;
}
