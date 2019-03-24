/*
This file contains the implementation of the forward and backward pass of
the exponent operation.
*/

#include "operations/exponentOperation.h"

#ifndef __OP_EXP_IMPL_INCLUDED__   
#define __OP_EXP_IMPL_INCLUDED__   

// Exponent Backprop
template <typename T>
void ExponentOperation<T>::backward(Matrix<T> grad) {
    this->t1->backward(grad*(this->t1->val.exp()));
}

template <typename T>
Tensor<T> ExponentOperation<T>::forward() {
    this->t3 = new Tensor<T>(this->t1->val.exp(), this);
    return *this->t3;
}

template <typename T>
Tensor<T>* ExponentOperation<T>::forwardPointer() {
    this->t3 = new Tensor<T>(this->t1->val.exp(), this);
    return this->t3;
}

#endif

