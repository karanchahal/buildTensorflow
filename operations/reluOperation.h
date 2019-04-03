/*
    This file defines the Sigmoid Operation class which represents the
    multiplication of two tensors.
*/

#include "operations/operation.h"

#ifndef __OP_RELU_INCLUDED__
#define __OP_RELU_INCLUDED__

template <typename T>
class ReluOperation : public Operation<T> {
    public:
   
    ReluOperation(Tensor<T> *t1) {
        this->t1 = t1;
    }
    void backward(Matrix<T> grad);

    Tensor<T> forward();

    Tensor<T>* forwardPointer();
};

#endif

