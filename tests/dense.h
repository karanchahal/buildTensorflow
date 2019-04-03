#include <gtest/gtest.h>
#include "layers/dense.h"
#include "tests/utils.h"

/*
    Tests that the gradient values are valid after doing 
    backpropation.
    TODO
*/
TEST(DENSE_LAYER_TESTS, SHAPE_CHECKS) {
    Dense<float> fc1(2,5,"sigmoid"); // input - 2, output should be 5
    Tensor<float>* x = new Tensor<float>({1,2},{1,2}); // put 1 by 2 tensor
    auto m = fc1.forward(x);
    m->backward();
}
