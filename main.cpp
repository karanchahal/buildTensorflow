#include "tensor.h" 
#include "operations_Impl.h"

template<typename T>
void printTensor(vector<T> &a) {
    for(auto i: a) {
        cout<<i<<" ";
    }
    cout<<endl;
}


void sigmoidTest() {
    Tensor<float> w0(vector<float>{2});
    Tensor<float> x0(vector<float>{-1});

    Tensor<float> w1(vector<float>{-3});
    Tensor<float> x1(vector<float>{-2});

    Tensor<float> w3(vector<float>{-3});


    Tensor<float> a = w0*x0;
    Tensor<float> b = w1*x1;
    Tensor<float> c = a + b;
    Tensor<float> d = w3+c;
    Tensor<float> e = Tensor<float>(vector<float>{-1});
    Tensor<float> f = d*e;
    Tensor<float> g = f.exp();
    Tensor<float> h = Tensor<float>(vector<float>{1});
    Tensor<float> i = g + h;
    Tensor<float> j = Tensor<float>(vector<float>{1});
    Tensor<float> k = j/i;

    k.backward(vector<float>{1});


    printTensor(w0.grad);
    printTensor(x0.grad);

    printTensor(w1.grad);
    printTensor(x1.grad);

    printTensor(w3.grad);
}


int main() {

    sigmoidTest();
}