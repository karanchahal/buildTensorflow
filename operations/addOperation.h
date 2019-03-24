/*
This file defines the AddOperation class which represents the addition of
two tensors.
*/

#include "operations/operation.h"

#ifndef __OP_ADD_INCLUDED__   
#define __OP_ADD_INCLUDED__   

template <typename T>
class AddOperation : public Operation<T> {
    public:
    
    AddOperation(Tensor<T> *t1, Tensor<T> *t2) {
        this->t1 = t1;
        this->t2 = t2;
    }

    void backward(Matrix<T> grad);

    Tensor<T> forward();

    Tensor<T>* forwardPointer();

};

#endif

