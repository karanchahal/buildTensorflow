/*
    This file defines the MultiplyOperation class which represents the
    multiplication of two tensors.
*/

#include "operations/operation.h"

#ifndef __OP_MULTIPLY_INCLUDED__
#define __OP_MULTIPLY_INCLUDED__

template <typename T>
class MultiplyOperation : public Operation<T> {
    public:
   
    MultiplyOperation(Tensor<T> *t1, Tensor<T> *t2) {
        this->t1 = t1;
        this->t2 = t2;
    }
    void backward(Matrix<T> grad);

    Tensor<T> forwardDeprecated();

    Tensor<T>* forward();
};

#endif

