/*
    This file tests the dense layer.
*/

#include <gtest/gtest.h>
#include "layers/dense.h"
#include "tests/utils.h"

/*
    Tests that the dense layer compiles successfully when shapes are correct
    and raises assert when shapes are not compatible.
*/
TEST(DENSE_LAYER_TESTS, SHAPE_CHECKS) {
    Dense<float> fc1(2,5); // input - 2, output should be 5
    Tensor<float>* x1 = new Tensor<float>({1,2},{1,2}); // put 1 by 2 tensor
    auto m = fc1.forward(x1); // should work fine

    delete m;

    ASSERT_DEATH({
       Tensor<float>* x2 = new Tensor<float>({1},{1}); // put 1 by 2 tensor
       Dense<float> fc1(2,5); // input - 2, output should be 5
       auto m = fc1.forward(x2); // should give error as dot product will not be compatible !
    }, "Shapes aren't compatible for dot product !");

}

/*
    Tests that the value being outputted by dense layer is correct or not:

    The output value should be given input x:
    y = activation(w*x + b)

    Here sigmoid is the default activation and glorot is the default weight initialisation.
*/
TEST(DENSE_LAYER_TESTS, CORRECTNESS_CHECK) {
    
    Dense<float> fc1(2,5);
    auto x = new Tensor<float>({1,2},{1,2}); 
    auto m = fc1.forward(x);
    auto w = fc1.weights->val;
    auto b = fc1.biases->val;

    auto expectedVal = matrixOps::sigmoid((x->val).dot(w) + b);

    ASSERT_TRUE(testUtils::isMatrixEqual(m->val, expectedVal));

    delete m;
}
