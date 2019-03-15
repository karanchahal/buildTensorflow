/*
This file defines the ExponentOperation class which represents the
exponentiation of a tensor.
*/

#include "operations/operation.h"

#ifndef __OP_EXP_INCLUDED__   
#define __OP_EXP_INCLUDED__   

template <typename T>
class ExponentOperation : public Operation<T> {
    public:
   
    ExponentOperation(Tensor<T> *t1) {
        this->t1 = t1;
    }

    void backward(Matrix<T> grad);

    Tensor<T> forward();

};

#endif

