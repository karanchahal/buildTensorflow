#include "operation.h"
#include "addOperation.h"
#include "multiplyOperation.h"


#ifndef __TENSOR_FLOAT_INCLUDED__   
#define __TENSOR_FLOAT_INCLUDED__   

class FloatTensor {
    public:
    float val; // value of tensor 
    float grad; // value of grad
    Operation *frontOp =NULL, *backOp =NULL;

    FloatTensor() {
        // default
    }

    FloatTensor(float val) {
        this->val = val;
    }

    FloatTensor(float val, Operation* op) {
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

    FloatTensor operator + (FloatTensor* two) { 
        this->frontOp = new AddOperation(this, two);
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

void AddOperation::backward(float grad) {
    // cout<<this->t2->val<<endl;
    // Distributing case: weherre gradients are backproped as is
    this->t1->backward(grad);
    this->t2->backward(grad);
}

FloatTensor AddOperation::compute() {
    return FloatTensor(this->t1->val+this->t2->val, this);
}

#endif