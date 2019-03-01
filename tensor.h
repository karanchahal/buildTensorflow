#include "operation.h"
#include "addOperation.h"
#include "multiplyOperation.h"
#include "divideOperation.h"
#include "exponentOperation.h"


#ifndef __TENSOR_FLOAT_INCLUDED__   
#define __TENSOR_FLOAT_INCLUDED__   

template<typename T>
void printTensor(vector<T> &a) {
    for(auto i: a) {
        cout<<i<<" ";
    }
    cout<<endl;
}


template <typename T>
class Tensor {
    public:
    vector<T> val; // value of tensor 
    vector<T> grad; // value of grad
    Operation<T> *frontOp =NULL, *backOp =NULL;

    Tensor() {
        // default
    }

    Tensor(const Tensor<T> *two) {
        this->val = two->val;
        this->backOp = two->backOp;
        this->frontOp = two->frontOp;
        this->grad = two->grad;
    }

    Tensor(vector<T> val) {
        this->val = val;
    }

    Tensor(vector<T> val, Operation<T>* op) {
        this->val = val;
        this->backOp = op;
    }

    bool isConstz(const Tensor<T> &one) {
        return true;
    }

    bool isConstz(Tensor<T> &one) {
        return false;
    }

    void backward(vector<T> grad) {
        // printTensor(this->val);
        this->grad = grad;
        if(this->backOp != NULL) {
            this->backOp->backward(grad);
        }
    }

    // Overloading Operations , one for const and one for actual variable
    // hence each operation has two overloads, except exp()
    Tensor<T> operator * (Tensor<T> &two) { 
        bool isConst = isConstz(*this);
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
        two.frontOp = this->frontOp;
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