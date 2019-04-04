/*
    This file contains the implementation of the forward and backward pass of
    the exponent operation.
*/

#include "operations/exponentOperation.h"

#ifndef __OP_EXP_IMPL_INCLUDED__   
#define __OP_EXP_IMPL_INCLUDED__   

/* 
    Backpropogation of the exponent operation. Example of a operation is as follows:
    F = x.exp() is forward propogation
    The gradients would be as follows:
    1. dF/dx = grad*x.exp() 
*/
template <typename T>
void ExponentOperation<T>::backward(Matrix<T> grad) {
    this->t1->backward(grad*(this->t1->val.exp()));
}

/* 
    Forward Propogation of the operation. Returns a tensor.

    TODO: Remove: See addition operation impl for more details
*/
template <typename T>
Tensor<T> ExponentOperation<T>::forward() {
    this->t3 = new Tensor<T>(this->t1->val.exp(), this);
    return *this->t3;
}

/* 
    Forward Propogation of the operation. Return pointer to the tensor.
*/
template <typename T>
Tensor<T>* ExponentOperation<T>::forwardPointer() {
    this->t3 = new Tensor<T>(this->t1->val.exp(), this);
    return this->t3;
}

#endif

