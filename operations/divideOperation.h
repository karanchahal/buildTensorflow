/*
    This file defines the DivideOperation class which represents the division of
    two tensors.
*/

#include "operations/operation.h"

#ifndef __OP_DIV_INCLUDED__   
#define __OP_DIV_INCLUDED__   

template <typename T>
class DivideOperation : public Operation<T> {
    public:
    
    DivideOperation(Tensor<T> *t1, Tensor<T> *t2) {
        this->t1 = t1;
        this->t2 = t2;
    }

    void backward(Matrix<T> grad);

    Tensor<T> forwardDeprecated();

    Tensor<T>* forward();

};

#endif

