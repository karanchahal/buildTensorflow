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

void sigmoidPointerTest() {

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
    
    auto grad = Matrix<float>({1},{1});
    k->backward(grad);


    cout<<w0->grad<<endl;
    cout<<x0->grad<<endl;

    cout<<w1->grad<<endl;
    cout<<x1->grad<<endl;

    cout<<w3->grad<<endl;
}

void updatedSigmoidtest() {
    Tensor<float>* w0 = new Tensor<float>({2},{1});
    Tensor<float>* x0= new Tensor<float>({-1},{1});

    Tensor<float>* w1= new Tensor<float>({-3},{1});
    Tensor<float>* x1= new Tensor<float>({-2},{1});

    Tensor<float>* w3= new Tensor<float>({-3},{1});
    
    auto a = tensorOps::multiply(w0,x0);
    auto b = tensorOps::multiply(w1,x1);
    auto c = tensorOps::add(a,b);
    auto d = tensorOps::add(w3,c);

    auto k = tensorOps::sigmoid(d);
    k->backward();

    cout<<w0->grad<<endl;
    cout<<x0->grad<<endl;

    cout<<w1->grad<<endl;
    cout<<x1->grad<<endl;

    cout<<w3->grad<<endl;

    delete k;
}

#include "optims/sgd.h"
#include "data/celsius2faranheit.h"

int main() {
    Celsius2Faranheit<float,float> dataset;
    dataset.create(5);
    for(auto i: dataset.dataset) {
        auto inp = i.first;
        auto tar = i.second;

        cout<<"Input: "<<inp<<" "<<"Target: "<<tar<<endl;
    }
}

