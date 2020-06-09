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
    virtual auto get_in_size() const -> uint32_t = 0;
    virtual auto get_out_size() const -> uint32_t = 0; 
  };

  struct MLP : DLModel {
    MLP(const std::vector<uint32_t>&);
    
    auto operator()(torch::Tensor&) -> torch::Tensor;

    auto forward(torch::Tensor&) -> torch::Tensor override;

    auto inline get_in_size() const -> uint32_t override {
      return insize;
    };

    auto inline get_out_size() const -> uint32_t override {
      return outsize;
    };

    std::vector<torch::nn::Linear> layers{};
    const uint32_t insize;
    const uint32_t outsize;
  };
  
  // declare a shorthand for model pointers
  typedef std::shared_ptr<DLModel> model_ptr; // can be std::dynamic_pointer_cast to mlp_ptr
  typedef std::shared_ptr<MLP> mlp_ptr;
  typedef std::unique_ptr<DLModel> u_torch_ptr;

  // make method for model_ptr
  auto make_MLP(const std::vector<uint32_t>&) -> model_ptr;

}
//TORCH_MODULE(MyNet); // add the name of the Value-ref'd class you want
#endif // 