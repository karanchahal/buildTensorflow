/*
    This file defines the ExponentOperation class which represents the
    exponentiation of a tensor.
*/

#include "operations/operation.h"

#ifndef __OP_LOG_INCLUDED__   
#define __OP_LOG_INCLUDED__   

template <typename T>
class LogOperation : public Operation<T> {
    public:
   
    LogOperation(Tensor<T> *t1) {
        this->t1 = t1;
    }

    void backward(Matrix<T> grad);

    Tensor<T> forwardDeprecated();

    Tensor<T>* forward();

};

#endif

