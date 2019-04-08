/*
    This file tests the tensor layer.
*/

#include "buildTensorflow.h"
#include <gtest/gtest.h>
#include "tests/utils.h"

using namespace std;

/*
    This test checks the functionality to create a tensor.
*/
TEST( TENSOR_TESTS, TensorCreation) {
    // Checks shape validation of Matrix
    ASSERT_DEATH({
        vector<int> a({1,2,3,4,5,6});
        vector<int> shape1({2,4});
        Tensor<int> m1(a,shape1);
    }, "Shape and size of vector are incompatible !");

    // testing for no asserts with various dimensions that can used in nd matrix
    vector<int> a({1,2,3,4,5,6});
    vector<int> shape1({2,3});
    vector<int> shape2({1,1,1,2,3});
    vector<int> shape3({2,3,1,1,1});
    Tensor<int> m1(a,shape1);
    m1 = Tensor<int>(a,shape2);
    m1 = Tensor<int>(a,shape3);
}

/*
    Tests that Tensor Operations yields the right result.
    A little redundant because Matrix Test validate results
    but good to have these tests in case of refactor
*/
TEST( TENSOR_TESTS, TensorAddOperations) {
    
    Tensor<int>* one = new Tensor<int>({1,2,3,4,5},{5});
    Tensor<int>* two = new Tensor<int>({1,2,3,4,5},{5});
    auto ans = tensorOps::add(one,two);
    Matrix<int> res({2,4,6,8,10},{5});

    ASSERT_TRUE(testUtils::isMatrixEqual(ans->val,res));

    // Clean up
    delete ans;
}


TEST( TENSOR_TESTS, TensorMultiplyOperations) {
    
    Tensor<int>* one = new Tensor<int>({1,2,3,4,5},{5});
    Tensor<int>* two = new Tensor<int>({1,2,3,4,5},{5});
    auto ans = tensorOps::multiply(one,two);
    Matrix<int> res({1,4,9,16,25},{5});

    ASSERT_TRUE(testUtils::isMatrixEqual(ans->val,res));

    // Clean up
    delete ans;
}

TEST( TENSOR_TESTS, TensorDivideOperations) {
    
    Tensor<int>* one = new Tensor<int>({5,6,10,4,1},{5});
    Tensor<int>* two = new Tensor<int>({1,3,2,2,1},{5});
    auto ans = tensorOps::divide(one,two);
    Matrix<int> res({5,2,5,2,1},{5});

    ASSERT_TRUE(testUtils::isMatrixEqual(ans->val,res));

    // Clean up
    delete ans;
}

/*
    This test checks the functionality of the sigmoid operation.
    Both front Prop and back Prop

    TODO: There is a small difference in the computation of the gradients 
    of the sigmoid operation with the formula sigmoid(1- sigmoid) and
    when it is done manually using the chain rule. Not yet known why this is.

    The difference is slight:

    With formula: 0.196611926
    Without formula: 0.196611971
*/
TEST( TENSOR_TESTS, TensorSigmoidOperations) {
    
    Tensor<float>* one = new Tensor<float>({1},{1});
    auto ans = tensorOps::sigmoid(one);
    Matrix<float> res({0.731058578}, {1});

    ASSERT_TRUE(testUtils::isMatrixEqual(ans->val,res)); // check front Propogation

    ans->backward();

    Matrix<float> resGrad({0.196611926}, {1});
    ASSERT_TRUE(testUtils::isMatrixEqual(one->grad,resGrad)); // check back Propogation

    // Clean up
    delete ans;
}

/*
    Test Computational Graph by checking Pointer Values of each
    tensor and operation for a barebones sigmoid function 
*/
TEST( TENSOR_TESTS, ComputationGraph) {
   
    Tensor<float>* w0 = new Tensor<float>({2},{1});
    Tensor<float>* x0= new Tensor<float>({-1},{1});

    Tensor<float>* w1= new Tensor<float>({-3},{1});
    Tensor<float>* x1= new Tensor<float>({-2},{1});

    Tensor<float>* w3= new Tensor<float>({-3},{1});
    
    auto a = tensorOps::multiply(w0,x0);
    auto b = tensorOps::multiply(w1,x1);
    auto c = tensorOps::add(a,b);
    auto d = tensorOps::add(w3,c);

    Tensor<float>* e = new Tensor<float>({-1}, {1});
    auto f = tensorOps::multiply(d,e);

    auto g = tensorOps::exp(f); // exponent

    Tensor<float>* h = new Tensor<float>({1}, {1});
    auto i = tensorOps::add(g,h);

    Tensor<float>* j = new Tensor<float>({1}, {1});
    auto k = tensorOps::divide(j,i);

    // Very tedious Test Coming Up now

    // Test k
    ASSERT_TRUE(k->frontOp == NULL);
    ASSERT_TRUE(k->backOp->t3 == k);

    ASSERT_TRUE(k->backOp->t1 == j);
    ASSERT_TRUE(k->backOp->t2 == i);

    // Test j
    ASSERT_TRUE(j->frontOp == k->backOp);
    ASSERT_TRUE(j->backOp == NULL);

    // Test i
    ASSERT_TRUE(i->frontOp == k->backOp);
    ASSERT_TRUE(i->backOp->t3 == i);

    ASSERT_TRUE(i->backOp->t1 == g);
    ASSERT_TRUE(i->backOp->t2 == h);
    
    // Test h
    ASSERT_TRUE(h->frontOp == i->backOp);
    ASSERT_TRUE(h->backOp == NULL);

    // Test g
    ASSERT_TRUE(g->frontOp == i->backOp);
    ASSERT_TRUE(g->backOp->t3 == g);

    ASSERT_TRUE(g->backOp->t1 == f);

    // Test f
    ASSERT_TRUE(f->frontOp == g->backOp);
    ASSERT_TRUE(f->backOp->t3 == f);

    ASSERT_TRUE(f->backOp->t1 == d);
    ASSERT_TRUE(f->backOp->t2 == e);

    // Test e
    ASSERT_TRUE(e->frontOp == f->backOp);
    ASSERT_TRUE(e->backOp == NULL);

    // Test d
    ASSERT_TRUE(d->frontOp == f->backOp);
    ASSERT_TRUE(d->backOp->t3 == d);

    ASSERT_TRUE(d->backOp->t1 == w3);
    ASSERT_TRUE(d->backOp->t2 == c);

    // Test w3
    ASSERT_TRUE(w3->frontOp == d->backOp);
    ASSERT_TRUE(w3->backOp == NULL);

    // Test c
    ASSERT_TRUE(c->frontOp == d->backOp);
    ASSERT_TRUE(c->backOp->t3 == c);

    ASSERT_TRUE(c->backOp->t1 == a);
    ASSERT_TRUE(c->backOp->t2 == b);

    // Test a
    ASSERT_TRUE(a->frontOp == c->backOp);
    ASSERT_TRUE(a->backOp->t3 == a);

    ASSERT_TRUE(a->backOp->t1 == w0);
    ASSERT_TRUE(a->backOp->t2 == x0);

    // Test b
    ASSERT_TRUE(b->frontOp == c->backOp);
    ASSERT_TRUE(b->backOp->t3 == b);

    ASSERT_TRUE(b->backOp->t1 == w1);
    ASSERT_TRUE(b->backOp->t2 == x1);

    // Test w0
    ASSERT_TRUE(w0->frontOp == a->backOp);
    ASSERT_TRUE(w0->backOp == NULL);

    // Test x0
    ASSERT_TRUE(x0->frontOp == a->backOp);
    ASSERT_TRUE(x0->backOp == NULL);

    // Test w1
    ASSERT_TRUE(w1->frontOp == b->backOp);
    ASSERT_TRUE(w1->backOp == NULL);

    // Test x1
    ASSERT_TRUE(x1->frontOp == b->backOp);
    ASSERT_TRUE(x1->backOp == NULL);

    // Clean up
    delete k;

}

/*
    Tests that the gradient values are valid after doing 
    backpropation.
*/
TEST(TENSOR_TESTS, BackwardPropogation) {
    Tensor<float>* w0 = new Tensor<float>({2},{1});
    Tensor<float>* x0= new Tensor<float>({-1},{1});

    Tensor<float>* w1= new Tensor<float>({-3},{1});
    Tensor<float>* x1= new Tensor<float>({-2},{1});

    Tensor<float>* w3= new Tensor<float>({-3},{1});
    
    auto a = tensorOps::multiply(w0,x0);
    auto b = tensorOps::multiply(w1,x1);
    auto c = tensorOps::add(a,b);
    auto d = tensorOps::add(w3,c);

    Tensor<float>* e = new Tensor<float>({-1}, {1});
    auto f = tensorOps::multiply(d,e);

    auto g = tensorOps::exp(f); // exponent

    Tensor<float>* h = new Tensor<float>({1}, {1});
    auto i = tensorOps::add(g,h);

    Tensor<float>* j = new Tensor<float>({1}, {1});
    auto k = tensorOps::divide(j,i);
    k->backward();

    // verify gradients
    auto res =  Matrix<float>({-0.196611971},{1});
    ASSERT_TRUE(testUtils::isMatrixEqual(w0->grad,res));

    res =  Matrix<float>({0.393223941},{1});
    ASSERT_TRUE(testUtils::isMatrixEqual(x0->grad,res));

    res =  Matrix<float>({-0.393223941},{1});
    ASSERT_TRUE(testUtils::isMatrixEqual(w1->grad,res));

    res =  Matrix<float>({-0.589835882},{1});
    ASSERT_TRUE(testUtils::isMatrixEqual(x1->grad,res));

    res =  Matrix<float>({0.196611971},{1});
    ASSERT_TRUE(testUtils::isMatrixEqual(w3->grad,res));

    // Clean up
    delete k;
}
