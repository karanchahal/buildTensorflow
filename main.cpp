#include "buildTensorflow.h"

void oldSigmoidTest() {

    Tensor<float> w0({2},{1});
    Tensor<float> x0({-1},{1});

    Tensor<float> w1({-3},{1});
    Tensor<float> x1({-2},{1});

    Tensor<float> w3({-3},{1});
    
    Tensor<float> a = w0*x0;
    Tensor<float> b = w1*x1;
    Tensor<float> c = a + b;
    Tensor<float> d = w3+c;
    Tensor<float> e({-1}, {1});
    Tensor<float> f = d*e;
    Tensor<float> g = f.exp();
    Tensor<float> h({1}, {1});
    Tensor<float> i = g + h;
    Tensor<float> j({1}, {1});
    Tensor<float> k = j/i;
    
    vector<float> vsl = {1};
    vector<int> sh = {1};
    auto grad = Matrix<float>(vsl,sh);
    k.backward(grad);


    cout<<w0.grad<<endl;
    cout<<x0.grad<<endl;

    cout<<w1.grad<<endl;
    cout<<x1.grad<<endl;

    cout<<w3.grad<<endl;
}

// WRONG BACKPROP: SOME ERROR WITH POINTERS AND OPERATION OVERLOADING.
void newSigmoidTest() {
    Tensor<float> w0({2},{1});
    Tensor<float> x0({-1},{1});

    Tensor<float> w1({-3},{1});
    Tensor<float> x1({-2},{1});

    Tensor<float> w3({-3},{1});
    Tensor<float> e({-1}, {1});
    Tensor<float> h({1}, {1});
    Tensor<float> j({1}, {1});

    Tensor<float> a = e*(w0*x0 + w1*x1 + w3);
    Tensor<float> k = j/(a.exp() + h);

    vector<float> vsl = {1};
    vector<int> sh = {1};
    auto grad = Matrix<float>(vsl,sh);
    k.backward(grad);


    cout<<w0.grad<<endl;
    cout<<x0.grad<<endl;

    cout<<w1.grad<<endl;
    cout<<x1.grad<<endl;

    cout<<w3.grad<<endl;
}

/*
API guide:

    1. Tensor(vector) - 1 D vector
    2. Tensor(vector, row, col) - 2 D vector
    Operations are add, dot, multiply, divide, exponent for tensors

    Matrix size will always be batch size, channels, height width
    Or batch size, embedding size, y

    Rules during simple add, sub. divide, multiply use broadcasting
    During matrix multiply, then 2D matrices can only be multiplied. 

    Cases will be multilayer perceptron, batch size, input vector
    layer weights would be input vector, output layer, hence add one to batch size dim
    The matrix multiply the uses two interior dims as 2D matrices as input
    Output will be 

*/

int main() {
    vector<int> a({1,2,3,4,5,6});
    vector<int> b({1,2,3,4,5,6,4,5,6});
    vector<int> shape1({2,3});
    vector<int> shape2({3,3});
    Matrix<int> m1(a,shape1);
    Matrix<int> m2(b,shape2);
    
  
    cout<<m1<<endl;
    cout<<m2<<endl;
    auto m3 = m1.dot(m2);
    cout<<m3<<endl;
    oldSigmoidTest();
    cout<<endl;
    newSigmoidTest();
}

