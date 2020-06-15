#include <iostream>
#include <torch/torch.h>
#include "models.h"
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

struct SimpleModel : torch::nn::Module {
    std::vector<torch::nn::Linear> layers;
    const uint32_t n_hidden = 10;
    
    SimpleModel(uint32_t n_in, uint32_t n_out) : 
        layers {
            register_module("linear_0", 
                    torch::nn::Linear(n_in, n_hidden)), 
            register_module("linear_1", 
                    torch::nn::Linear(n_hidden, n_out))
        }
    { std::cout << "SimpleModel constructed! \n";}

    auto forward(const torch::Tensor& x) -> torch::Tensor {
        auto l1 =  layers[0]->forward(x);
        auto l2 =   torch::relu(l1);
        return layers[1]->forward(l2);
    }
};

template<typename T>
auto test_function(const std::shared_ptr<T>& mlp) -> float {
    auto a = torch::rand({100,2});
    return torch::mean( mlp->forward(a)).template item<float>();
}

using namespace torch::indexing;

auto main(int, char**) -> int {

    // check if anything gets copied
    auto a = torch::ones({1000, 2}); // [batch, n_in]
    auto b = a;
    auto c(a);
    
    a[0][1] = 42;
    std::cout << "b[0,1]== " << b[0][1].item<float>() << ", c[0][1] == " << c[0][1].item<float>() << '\n'; // compare with forward(torch::Tensor&)
    // *********************************************************************************
    auto model_ptr = std::make_shared<SimpleModel>(2,2);
    auto d = model_ptr->forward(a);
    std::cout << torch::mean(d).item<float>() << '\n';
    auto mean = test_function(model_ptr);
    std::cout << mean << '\n';
    return EXIT_SUCCESS;
}
