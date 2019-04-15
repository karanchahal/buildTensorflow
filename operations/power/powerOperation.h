/*
    This file defines the PowerOperation class which represents the
    exponentiation of a tensor with a scalar.
*/

#include "operations/operation.h"

#ifndef __OP_POWER_INCLUDED__
#define __OP_POWER_INCLUDED__

template <typename T>
class PowerOperation : public Operation<T> {
    public:
    T pow;

    PowerOperation(Tensor<T> *t1, T pow) {
        this->t1 = t1;
        this->pow = pow;
    }
    void backward(Matrix<T> grad);

    Tensor<T> forwardDeprecated();

    Tensor<T>* forward();
};

#endif

