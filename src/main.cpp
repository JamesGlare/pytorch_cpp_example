#include <iostream>
#include <thread>
#include <torch/torch.h>
#include "models.h"
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
    const float target_loss = 0.01;
    const uint32_t max_layers = 15;
    const uint32_t min_layer_size =2;
    const uint32_t max_layer_size = 1000;
    const uint32_t max_rounds = 10;
    const uint32_t population = 10;
    const double pchange = 0.1;
    const double std = 0.1;
    const double noise_std = 0.1;
    const uint32_t n_dim_in = 5;
    const uint32_t n_dim_out = 5;
    

    auto models = evolve::evolve(target_loss, max_layers, min_layer_size, 
                                 max_layer_size, max_rounds, population, pchange, std,
                                 n_dim_in, n_dim_out,
                                 [noise_std](const torch::Tensor& t) 
                                 { return utils::noisy_sinus(t, noise_std); },
                                 models::make_MLP);
    std::for_each(models.begin(), models.end(), 
                 [](models::model_ptr& model){ utils::explain(*model); });
    return EXIT_SUCCESS;
}
