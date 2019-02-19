#include "operation.h"
#include "addOperation.h"
#include "multiplyOperation.h"
#include "divideOperation.h"
#include "exponentOperation.h"


#ifndef __TENSOR_FLOAT_INCLUDED__   
#define __TENSOR_FLOAT_INCLUDED__   

template <typename T>
class Tensor {
    public:
    vector<T> val; // value of tensor 
    vector<T> grad; // value of grad
    Operation<T> *frontOp =NULL, *backOp =NULL;

    Tensor() {
        // default
    }

    Tensor(vector<T> val) {
        this->val = val;
    }

    Tensor(vector<T> val, Operation<T>* op) {
        this->val = val;
        this->backOp = op;
    }

    void backward(vector<T> grad) {
        this->grad = grad;
        if(this->backOp != NULL) {
            this->backOp->backward(grad);
        }
    }

    Tensor<T> operator * (Tensor<T> &two) { 
        this->frontOp = new MultiplyOperation<T>(this, &two);
        return this->frontOp->forward();
    }

    Tensor<T> operator + (Tensor<T> two) { 
        this->frontOp = new AddOperation<T>(this, &two);
        return this->frontOp->forward();
    }

    Tensor<T> operator / (Tensor<T> &two) { 
        this->frontOp = new DivideOperation<T>(this, &two);
        return this->frontOp->forward();
    }

    Tensor<T> exp() { 
        this->frontOp = new ExponentOperation<T>(this);
        return this->frontOp->forward();
    }
};


#endif