#include "models.h"

namespace models{

    MLP::MLP(const std::vector<size_t>& layer_widths) : 
        insize{layer_widths.front()}, outsize{layer_widths.back()}
    {
        size_t l = 0;
        for(auto it = layer_widths.cbegin(); 
                it < layer_widths.cend() -1; ++it) {
            const auto n = *it, m = *(it+1);
            layers.emplace_back(
                        register_module("linear_" + std::to_string(l), 
                            torch::nn::Linear(n, m)));
            ++l;
        }
    }
    auto MLP::operator()(torch::Tensor& input) -> torch::Tensor {
        return this->forward(input);
    }
    // 
    auto MLP::forward(torch::Tensor& input) -> torch::Tensor {
        if (layers.size() == 0)
            return input;

        auto output = torch::relu(layers.front()->forward(input));
        for(auto it = layers.begin()+1; it < layers.end(); ++it){
            output = torch::relu((*it)->forward(output));
        }
        return output;
    }

    auto make_MLP(const std::vector<size_t>& layer_widths) -> torch_ptr {
        return std::make_shared<MLP>(layer_widths);
    }
}