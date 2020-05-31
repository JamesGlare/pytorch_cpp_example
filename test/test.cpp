#include <gtest/gtest.h>
#include "models.h"
/*

*/
namespace unittests {
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