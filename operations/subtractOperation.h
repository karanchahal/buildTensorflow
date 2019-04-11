/*
    This file defines the SubtractOperation class which represents the subtraction of
    two tensors.
*/

#include "operations/operation.h"

#ifndef __OP_SUBTRACT_INCLUDED__
#define __OP_SUBTRACT_INCLUDED__

template <typename T>
class SubtractOperation : public Operation<T> {
    public:
    
    SubtractOperation(Tensor<T> *t1, Tensor<T> *t2) {
        this->t1 = t1;
        this->t2 = t2;
    }

    void backward(Matrix<T> grad);

    Tensor<T>* forward();

    Tensor<T> forwardDeprecated();

};

#endif
