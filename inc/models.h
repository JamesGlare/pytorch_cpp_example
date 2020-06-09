#ifndef  __MODELS__
#define __MODELS__
#include <vector>
#include <torch/torch.h>
#include <memory>

namespace models {
  struct DLModel : torch::nn::Module{
    /* abstract virtual base class
     */
    virtual auto forward(torch::Tensor&) -> torch::Tensor = 0;
    virtual auto get_in_size() -> const size_t = 0;
    virtual auto get_out_size() -> const size_t = 0; 
  };

  // declare a shorthand for model pointers
  typedef std::shared_ptr<DLModel> model_ptr;
  typedef std::unique_ptr<DLModel> u_torch_ptr;

  struct MLP : DLModel {
    MLP(const std::vector<size_t>&);
    
    auto operator()(torch::Tensor&) -> torch::Tensor;

    auto forward(torch::Tensor&) -> torch::Tensor override;

    auto inline get_in_size() -> const size_t override {
      return insize;
    };

    auto inline get_out_size() -> const size_t override {
      return outsize;
    };

    std::vector<torch::nn::Linear> layers{};
    const size_t insize;
    const size_t outsize;
  };

  model_ptr make_MLP(const std::vector<size_t>&);

}
//TORCH_MODULE(MyNet); // add the name of the Value-ref'd class you want
#endif // 