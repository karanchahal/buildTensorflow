/*
    This file defines the Sigmoid Operation class which represents the
    multiplication of two tensors.
*/

#include "operations/operation.h"

#ifndef __OP_SOFTMAX_INCLUDED__
#define __OP_SOFTMAX_INCLUDED__

template <typename T>
class SoftmaxOperation : public Operation<T> {
    public:
   
    SoftmaxOperation(Tensor<T> *t1) {
        this->t1 = t1;
    }
    void backward(Matrix<T> grad);

    Tensor<T> forwardDeprecated();

    Tensor<T>* forward();
};

#endif

