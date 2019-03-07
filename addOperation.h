/*
This file defines the AddOperation class which represents the addition of
two tensors.
*/

#include "operation.h"

#ifndef __OP_ADD_INCLUDED__   
#define __OP_ADD_INCLUDED__   

template <typename T>
class AddOperation : public Operation<T> {
    public:
        Tensor<T> *t1, *t2; // Input tensors
        Tensor<T> *t3; // Output tensor
    
    AddOperation(Tensor<T> *t1, Tensor<T> *t2) {
        this->t1 = t1;
        this->t2 = t2;
    }

    void backward(vector<T> grad);
    Tensor<T> forward();
};
#endif
