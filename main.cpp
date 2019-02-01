#include<iostream>

using namespace std; 

// graph has nodes and egdes
// our edges will be tensors
// our nodes will be operations

class Operation {

};

// class MultiplyOperation {

// };

// class AddOperation {

// };

class Tensor {

};

// after handling basic numbers
// we come to matrices. 
// I can also you how to do really fast matrix multiplcations using CUDA , put it on the GPU

// train a network hopefully. Maybe MNIST

// build python warppaers around out C++ api

class MultiplyOperation {


    public:
        FloatTensor t1, t2; // generally an operation had two operands
    
    void backward(Gradient grad) {
        // switcher,
        t1.backward(grad*t2.val);
        t1.backward(grad*t1.val);
    }

    void compute() {
        return new Floattensor(t1.val*t2.val, backwardOperation=this)
    }
}


class AddOperation {

    public:
        FloatTensor t1, t2; // generally an operation had two operands
    void backward(Gradient grad) {
        // distributor
        t1.backward(grad);
        t1.backward(grad);
    }
}

class FloatTensor {
    public:
    float val; // value of tensor 
    float grad; // value of grad

    FloatTensor() {
        // default
    }

    FloatTensor(float val) {
        this->val = val;
    }

 

    void backward(Gradient grad) {
        this->grad = grad;
        // this->val = this->val -learning_rate*grad (gradient decent)

        if(this->backwardOperation != NULL)
            this->backwardOperation.backward(grad); // this would be None so stop here
    }

    FloatTensor* operator * (FloatTensor* two) { 
        this->forwardOperation = MultiplyOperation(this, two)

        return this->forwardOperation.compute();
    }
};


// vector Tensor 

int main() {
    // and autodiff library
    FloatTensor* one = new FloatTensor(2);
    FloatTensor* two = new FloatTensor(4);

    FloatTensor* three = one*two;
    three.backward(1);
}