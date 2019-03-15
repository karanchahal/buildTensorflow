#include<iostream>
#include <gtest/gtest.h>
#include "tensor/matrix.h"

using namespace std;


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

template<typename T>
bool isMatrixEqual(Matrix<T> &lhs, Matrix<T> &rhs) {
    int n = lhs.shape.size();
    int m = rhs.shape.size();

    if(n != m) {
        return false;
    }

    for(int i = 0; i<n;i++) {
        if(lhs.shape[i] != rhs.shape[i]) {
            return false;
        }
    }

    n = lhs.val.size();

    for(int i = 0;i<n;i++) {
        if(lhs.val[i] != rhs.val[i]) {
            return false;
        }
    }



    return true;
}

TEST( MATRIX_TESTS, MatrixOperationAdditionCheck) {
    
    vector<int> a({1,2,3});
    vector<int> shape1({1,3});
    vector<int> b({1,2,3});
    vector<int> shape2({1,3});
    Matrix<int> m1(a,shape1);
    Matrix<int> m2(b,shape2);
    auto ans = m1+m2;
    Matrix<int> res({2,4,6},{1,3});

    ASSERT_TRUE(isMatrixEqual<int>(ans,res));

}

TEST( MATRIX_TESTS, MatrixOperationMultiplicationCheck) {

    vector<int> a({1,2,3});
    vector<int> shape1({1,3});
    vector<int> b({1,2,3});
    vector<int> shape2({1,3});
    Matrix<int> m1(a,shape1);
    Matrix<int> m2(b,shape2);
    auto ans = m1*m2;
    Matrix<int> res({1,4,9},{1,3});

    ASSERT_TRUE(isMatrixEqual<int>(ans,res));
}

TEST( MATRIX_TESTS, MatrixOperationDivisionCheck) {
    vector<int> a({9,4,3});
    vector<int> shape1({1,3});
    vector<int> b({1,2,3});
    vector<int> shape2({1,3});
    Matrix<int> m1(a,shape1);
    Matrix<int> m2(b,shape2);
    auto ans = m1/m2;
    Matrix<int> res({9,2,1},{1,3});

    ASSERT_TRUE(isMatrixEqual<int>(ans,res));
}

TEST( MATRIX_TESTS, MatrixOperationPowerCheck) {
    vector<int> a({1,2,3});
    vector<int> shape1({1,3});
    Matrix<int> m1(a,shape1);
    auto ans = m1^2;
    Matrix<int> res({1,4,9},{1,3});
    ASSERT_TRUE(isMatrixEqual<int>(ans,res));
}

TEST( MATRIX_TESTS, MatrixOperationExponentCheck) {
    vector<float> a({1,2,3});
    vector<int> shape1({1,3});
    Matrix<float> m1(a,shape1);
    auto ans = m1.exp();
    Matrix<float> res({ (float)exp(1),  (float)exp(2),  (float)exp(3)},{1,3});

    ASSERT_TRUE(isMatrixEqual<float>(ans,res));
}

TEST( MATRIX_TESTS, MatrixOperationDotProductCheck) {
    vector<int> a({1,2,3,1,2,3});
    vector<int> shape1({2,1,3});
    vector<int> b({1,2,3});
    vector<int> shape2({3,1});
    Matrix<int> m1(a,shape1);
    Matrix<int> m2(b,shape2);
    auto ans = m1.dot(m2);
    Matrix<int> res({14,14},{2,1,1});

    ASSERT_TRUE(isMatrixEqual(ans,res));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

