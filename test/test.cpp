#include <gtest/gtest.h>
#include "models.h"
#include "utils.h"
/*

*/
namespace unittests {
    TEST(test_pytorch, sort_test){
                                //  0 1 2 3 4 5 6 7 8 9
        std::vector<uint32_t> order{3,8,1,2,9,6,7,4,5,0};
        std::vector<uint32_t> result{9,2,3,0,7,8,5,6,1,4};
        std::vector<uint32_t> indices(order.size());
        std::iota(indices.begin(), indices.end(), 0 );
        utils::sort_by(indices, order);
        ASSERT_EQ(indices, result);
    }
    TEST(test_pytorch, mutation_std_test){
        uint32_t n = 100;
        double std = 0.05;
        uint32_t min = std::max((1.-std)*n,0.);
        uint32_t max = (1.+std)*n;
        ASSERT_GE(max, n);
        ASSERT_LE(min, n);
    }
    TEST(test_pytorch, mlp_depth_test){
        auto mlp = 	std::dynamic_pointer_cast<models::MLP>(
                        models::make_MLP({2, 100, 2})
                    );
        ASSERT_EQ(mlp->layers.size(), 2);
    }
    
    TEST(test_pytorch, mlp_architecture_test)
    {
        auto mlp = models::make_MLP({2, 100, 2});
        auto parameter_dict = mlp->named_parameters();
        EXPECT_EQ(
            parameter_dict["linear_0.weight"].size(0), 100);
        EXPECT_EQ(
            parameter_dict["linear_0.weight"].size(1), 2);    
        EXPECT_EQ(
            parameter_dict["linear_0.bias"].size(0), 100);
        EXPECT_EQ(
            parameter_dict["linear_1.weight"].size(0), 2);
        EXPECT_EQ(
            parameter_dict["linear_1.weight"].size(1), 100);
        EXPECT_EQ(
            parameter_dict["linear_1.bias"].size(0), 2);
    }
}

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}