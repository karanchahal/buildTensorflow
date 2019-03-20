#include "buildTensorflow.h"
#include <gtest/gtest.h>

using namespace std;


TEST( TENSOR_TESTS, TensorCreation) {
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

TEST( TENSOR_TESTS, TensorBackOp) {
    
    vector<int> a({1,2,3});
    vector<int> shape1({1,3});
    vector<int> b({1,2,3});
    vector<int> shape2({1,3});
    Matrix<int> m1(a,shape1);
    Matrix<int> m2(b,shape2);
    auto ans = m1+m2;
    Matrix<int> res({2,4,6},{1,3});

    // ASSERT_TRUE(isMatrixEqual<int>(ans,res));

}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

