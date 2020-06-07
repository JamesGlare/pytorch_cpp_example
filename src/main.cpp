#include <iostream>
#include <torch/torch.h>
#include "models.h"
#include "utils.h"
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
using namespace torch::indexing;

auto train_model(models::torch_ptr model,
                 const std::string& save_path,
                 const uint32_t n_sample,
                 const uint32_t n_batch,
                 const double lr,
                 const uint32_t log_steps = 100
                 ) -> void {
    model->train();
    torch::optim::Adam optimizer(model->parameters(), lr);
    for(uint32_t batch_idx = 0; batch_idx < n_sample/n_batch; ++batch_idx){
       auto batch_xy = utils::generate_xy(n_batch, model->get_in_size());
       optimizer.zero_grad();
       auto prediction = model->forward(batch_xy.first);
       auto loss = torch::mse_loss(prediction, batch_xy.second);
       loss.backward(); 
       optimizer.step();
       if (batch_idx % log_steps == 0) {
           std::cout << "Batch: " << batch_idx
                     << " | Loss: " << loss.item<float>() << '\n';
       }
    }
}
auto test_model(models::torch_ptr model,
                const uint32_t n_sample,
                const uint32_t n_batch,
                std::ostream& write_to
                ) -> void {
    model->eval();
    write_to << "Test results: " << '\n';
    for (uint32_t batch_idx = 0; batch_idx< n_sample/n_sample; ++batch_idx) {
        auto batch_xy = utils::generate_xy(n_batch, model->get_in_size());
        auto prediction = model->forward(batch_xy.first);
        auto loss = torch::mse_loss(prediction, batch_xy.second);

        for (uint32_t b=0; b < n_batch; ++b) {
            for (uint32_t d=0; d< model->get_in_size(); ++d ){
                write_to << batch_xy.first[b][d].item<float>()  << "\t" 
                         << batch_xy.second[b][d].item<float>() << '\t'
                         << prediction[b][d].item<float>()      << '\n';
            }
        }
    }
}

auto main(int, char**) -> int {
    auto mynet = models::MLP({2, 100, 2}); //models::make_MLP({2, 100, 2}); // you want to have your net class on the heap -> threads etc
    utils::explain(mynet);

    // check if anything gets copied
    auto a = torch::ones({1000, 2}); // [batch, n_in]
    auto b = a;
    auto c(a);
    
    a[0][1] = 42;
    std::cout << "b[0,1]== " << b[0][1].item<float>() << ", c[0][1] == " << c[0][1].item<float>() << '\n'; // compare with forward(torch::Tensor&)
    // *********************************************************************************
    const std::string save_path{"./net.pt"};
    
    const uint32_t n_sample = 10000;
    const uint32_t n_batch = 32;
    const double lr = 1e-3;
    auto model = models::make_MLP({2,200,2});

    train_model(model, save_path, n_sample, n_batch, lr);
    test_model(model, 10, 10, std::cout);
    return EXIT_SUCCESS;
}
