#include<iostream>

using namespace std; 

class FloatTensor;

class MultiplyOperation {


    public:
        FloatTensor *t1, *t2; // generally an operation had two operands
    
    MultiplyOperation(FloatTensor *t1, FloatTensor *t2) {
        this->t1 = t1;
        this->t2 = t2;
    }
    void backward(float grad);

    FloatTensor compute();
};


class FloatTensor {
    public:
    float val; // value of tensor 
    float grad; // value of grad
    MultiplyOperation *frontOp =NULL, *backOp =NULL;

    FloatTensor() {
        // default
    }

    FloatTensor(float val) {
        this->val = val;
    }

    FloatTensor(float val, MultiplyOperation* op) {
        this->val = val;
        this->backOp = op;
    }

    void backward(float grad) {
        this->grad = grad;
        if(this->backOp != NULL) {
            this->backOp->backward(grad);
        }
    }


    FloatTensor operator * (FloatTensor* two) { 
        this->frontOp = new MultiplyOperation(this, two);
        return this->frontOp->compute();
    }
};


void MultiplyOperation::backward(float grad) {
    // cout<<this->t2->val<<endl;
    // Swithing case: weherre gradients are multiplied with the opposite tensor
    this->t1->backward(grad*this->t2->val);
    this->t2->backward(grad*this->t1->val);
}

FloatTensor MultiplyOperation::compute() {
    return FloatTensor(this->t1->val*this->t2->val, this);
}


// vector Tensor 

int main() {

    FloatTensor one(2); // heap declaration
    FloatTensor two(4);
    
    FloatTensor three = one*(&two);
    three.backward(1);

    cout<<three.grad<<endl; // 1
    cout<<one.grad<<endl; // 1*4 = 4
    cout<<two.grad<<endl; // 1*2 = 2
}