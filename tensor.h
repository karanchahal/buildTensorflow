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

    Tensor(const Tensor<T>* two) {
        this->backOp = two->backOp;
        this->frontOp = two->frontOp;
        this->val = two->val;
        this->grad = two->grad;
    }

    void backward(vector<T> grad) {
        this->grad = grad;
        if(this->backOp != NULL) {
            this->backOp->backward(grad);
        }
    }
    // Two overloaded functions for each operator to get const temporaries and actual tensors

    Tensor<T> operator * (Tensor<T> &two) { 
        this->frontOp = new MultiplyOperation<T>(this, &two);
        two.frontOp = this->frontOp;
        return this->frontOp->forward();
    }

    Tensor<T> operator * (const Tensor<T> &two) { 
        Tensor<T>* temp = new Tensor<T>(&two);
        this->frontOp = new MultiplyOperation<T>(this, temp);
        temp->frontOp = this->frontOp;
        return this->frontOp->forward();
    }

    // Tensor<T> operator * (Tensor<T> &two) { 
    //     this->frontOp = new MultiplyOperation<T>(this, &two);
    //     two.frontOp = this->frontOp;
    //     return this->frontOp->forward();
    // }

    Tensor<T> operator + (Tensor<T> &two) { 
        this->frontOp = new AddOperation<T>(this, &two);
        two.frontOp = this->frontOp;
        return this->frontOp->forward();
    }

    Tensor<T> operator + (const Tensor<T> &two) { 
        Tensor<T>* temp = new Tensor<T>(&two);
        this->frontOp = new AddOperation<T>(this, temp);
        temp->frontOp = this->frontOp;
        return this->frontOp->forward();
    }

    Tensor<T> operator / (Tensor<T> &two) { 
        this->frontOp = new DivideOperation<T>(this, &two);
        return this->frontOp->forward();
    }

    Tensor<T> operator / (const Tensor<T> &two) { 
        Tensor<T>* temp = new Tensor<T>(&two);
        this->frontOp = new DivideOperation<T>(this, temp);
        temp->frontOp = this->frontOp;
        return this->frontOp->forward();
    }

    Tensor<T> exp() { 
        this->frontOp = new ExponentOperation<T>(this);
        return this->frontOp->forward();
    }
};


#endif