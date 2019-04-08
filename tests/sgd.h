/*
    This file tests the SGD Optimizer layer.
*/

#include <gtest/gtest.h>
#include "optims/sgd.h"
#include "tests/utils.h"
#include "overloads/tensor.h"

/*
    Tests that the optimizer layer gets all the tensors that need to be updated.
*/
TEST(SGD_OPTIM_TESTS, TENSOR_UPDATE_CHECK) {
    Tensor<float>* a = new Tensor<float>({2},{1});
    Tensor<float>* b = new Tensor<float>({4},{1});
    auto c = tensorOps::add(a,b);
    Tensor<float>* d = new Tensor<float>({3},{1});

    auto e = tensorOps::multiply(c,d);
    e->backward();

    SGD<float> sgd(0.1);
    // get all paramters/tensors that need to be updated wrt to e
    sgd.getParams(e);
    unordered_set<Tensor<float>*> expected_res = {a,b,c,d};
    ASSERT_TRUE(sgd.params == expected_res);

    // Clean up
    delete e;
}

/*
    Tests that the tensor values are updated according to gradient values and learning rate
*/
TEST(SGD_OPTIM_TESTS, SGD_STEP_CHECK) {
    Tensor<float>* a = new Tensor<float>({2},{1});
    Tensor<float>* b = new Tensor<float>({4},{1});
    auto c = tensorOps::add(a,b);
    Tensor<float>* d = new Tensor<float>({3},{1});

    auto e = tensorOps::multiply(c,d);
    e->backward();

    SGD<float> sgd(1);
    // get all paramters/tensors that need to be updated wrt to e
    sgd.minimise(e);

    ASSERT_TRUE(a->val.val[0] == -1); // update = 2 - 1*3
    ASSERT_TRUE(b->val.val[0] == 1); // update = 4 - 1*3
    ASSERT_TRUE(d->val.val[0] == -3); // update = 3 -1*6

    // Clean up
    delete e;
}
