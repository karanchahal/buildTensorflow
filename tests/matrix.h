/*
    This file tests the matrix layer.
*/

#include <iostream>
#include <gtest/gtest.h>
#include "types/matrix.h"
#include "overloads/matrix.h"
#include "tests/utils.h"

using namespace std;

/*
    This test tests the validity of matrix creation
*/
TEST( MATRIX_TESTS, MatrixCreationShapeValidation) {
    // Checks shape validation of Matrix
    ASSERT_DEATH({
        vector<int> a({1,2,3,4,5,6});
        vector<int> shape1({2,4});
        Matrix<int> m1(a,shape1);
    }, "Shape and size of vector are incompatible !");

    // testing for no asserts with various dimensions that can used in nd matrix
    vector<int> a({1,2,3,4,5,6});
    vector<int> shape1({2,3});
    vector<int> shape2({1,1,1,2,3});
    vector<int> shape3({2,3,1,1,1});
    Matrix<int> m1(a,shape1);
    m1 = Matrix<int>(a,shape2);
    m1 = Matrix<int>(a,shape3);
}

/*
    This test tests the shape validation function of the matrix
*/
TEST( MATRIX_TESTS, MatrixOperationShapeValidation) {
    // Checks for matrix addition
    ASSERT_DEATH({
        vector<int> a({1,2,3,4,5,6});
        vector<int> shape1({2,3});
        vector<int> b({1,2,3,4,5,6});
        vector<int> shape2({3,2});
        Matrix<int> m1(a,shape1);
        Matrix<int> m2(b,shape2);
        auto ans = m1+m2;
    }, "Shapes aren't compatible for addition !");

    // multiplication
    // Checks for matrix addition
    ASSERT_DEATH({
        vector<int> a({1,2,3,4,5,6});
        vector<int> shape1({2,3});
        vector<int> b({1,2,3,4,5,6});
        vector<int> shape2({3,2});
        Matrix<int> m1(a,shape1);
        Matrix<int> m2(b,shape2);
        auto ans = m1*m2;
    }, "Shapes aren't compatible for multiplication !");

    // division
    // Checks for matrix addition
    ASSERT_DEATH({
        vector<int> a({1,2,3,4,5,6});
        vector<int> shape1({2,3});
        vector<int> b({1,2,3,4,5,6});
        vector<int> shape2({3,2});
        Matrix<int> m1(a,shape1);
        Matrix<int> m2(b,shape2);
        auto ans = m1/m2;
    }, "Shapes aren't compatible for division !");

    // dot product shape checks
    ASSERT_DEATH({
        vector<int> a({1,2,3,4,5,6});
        vector<int> shape1({2,3});
        vector<int> b({1,2,3,4,5,6});
        vector<int> shape2({2,3});
        Matrix<int> m1(a,shape1);
        Matrix<int> m2(b,shape2);
        auto ans = m1.dot(m2);
    }, "Shapes aren't compatible for dot product !");

    ASSERT_DEATH({
        vector<int> a({1,2,3,4,5,6});
        vector<int> shape1({2,3});
        vector<int> b({1,2,3,4,5,6});
        vector<int> shape2({3,2,1});
        Matrix<int> m1(a,shape1);
        Matrix<int> m2(b,shape2);
        auto ans = m1.dot(m2);
    }, "Shapes aren't compatible for dot product !");
}

/*
    This test tests the accuracy of the addition operation between 2 matrices
*/
TEST( MATRIX_TESTS, MatrixOperationAdditionCheck) {
    
    vector<int> a({1,2,3});
    vector<int> shape1({1,3});
    vector<int> b({1,2,3});
    vector<int> shape2({1,3});
    Matrix<int> m1(a,shape1);
    Matrix<int> m2(b,shape2);
    auto ans = m1+m2;
    Matrix<int> res({2,4,6},{1,3});

    ASSERT_TRUE(testUtils::isMatrixEqual<int>(ans,res));

}

/*
    This test tests the accuracy of the multiplication operation between 2 matrices
*/
TEST( MATRIX_TESTS, MatrixOperationMultiplicationCheck) {

    vector<int> a({1,2,3});
    vector<int> shape1({1,3});
    vector<int> b({1,2,3});
    vector<int> shape2({1,3});
    Matrix<int> m1(a,shape1);
    Matrix<int> m2(b,shape2);
    auto ans = m1*m2;
    Matrix<int> res({1,4,9},{1,3});

    ASSERT_TRUE(testUtils::isMatrixEqual<int>(ans,res));
}

/*
    This test tests the accuracy of the power operation between a matrix and a scalar
*/
TEST( MATRIX_TESTS, MatrixOperationPowerCheck) {

    vector<int> a({1,2,3});
    vector<int> shape1({1,3});
    Matrix<int> m1(a,shape1);
    int pow = 3;
    auto ans = m1^pow; // Checking barebones operation
    Matrix<int> res({1,8,27},{1,3});

    ASSERT_TRUE(testUtils::isMatrixEqual<int>(ans,res));

    Matrix<int> m2({1,2,3},{1,3});
    pow = 2;
    Matrix<int> res2({1,4,9},{1,3});
    auto ans2 = matrixOps::power(m2,pow); // Checking wrapper function

    ASSERT_TRUE(testUtils::isMatrixEqual<int>(ans2,res2));
}

/*
    This test tests the accuracy of the division operation between 2 matrices
*/
TEST( MATRIX_TESTS, MatrixOperationDivisionCheck) {
    vector<int> a({9,4,3});
    vector<int> shape1({1,3});
    vector<int> b({1,2,3});
    vector<int> shape2({1,3});
    Matrix<int> m1(a,shape1);
    Matrix<int> m2(b,shape2);
    auto ans = m1/m2;
    Matrix<int> res({9,2,1},{1,3});

    ASSERT_TRUE(testUtils::isMatrixEqual<int>(ans,res));
}


/*
    This test tests the accuracy of the exponent operation.
*/
TEST( MATRIX_TESTS, MatrixOperationExponentCheck) {
    vector<float> a({1,2,3});
    vector<int> shape1({1,3});
    Matrix<float> m1(a,shape1);
    auto ans = m1.exp();
    Matrix<float> res({ (float)exp(1),  (float)exp(2),  (float)exp(3)},{1,3});

    ASSERT_TRUE(testUtils::isMatrixEqual<float>(ans,res));
}

/*
    This test tests the accuracy of the dot product operation between 2 matrices
*/
TEST( MATRIX_TESTS, MatrixOperationDotProductCheck) {
    vector<int> a({1,2,3,1,2,3});
    vector<int> shape1({2,1,3});
    vector<int> b({1,2,3});
    vector<int> shape2({3,1});
    Matrix<int> m1(a,shape1);
    Matrix<int> m2(b,shape2);
    auto ans = m1.dot(m2);
    Matrix<int> res({14,14},{2,1,1});

    ASSERT_TRUE(testUtils::isMatrixEqual(ans,res));
}

/*
    This test tests the accuracy of the sigmoid operation.
*/
TEST( MATRIX_TESTS, MatrixOperationSigmoidCheck) {

    Matrix<float> w0({2},{1});
    Matrix<float> x0({-1},{1});

    Matrix<float> w1({-3},{1});
    Matrix<float> x1({-2},{1});

    Matrix<float> w3({-3},{1});
    Matrix<float> w4({1},{1});
    auto x = w0*x0 + w1*x1 + w3;
    auto y = matrixOps::sigmoid(x);
    Matrix<float> res({0.731058578}, {1});
    
    ASSERT_TRUE(testUtils::isMatrixEqual(y,res));
}
