/*
    This file contains the implementation of the forward and backward pass of
    the multiply operation.
*/

#include "operations/multiply/multiplyOperation.h"

#ifndef __OP_IMPL_MULTIPLY_INCLUDED__
#define __OP_IMPL_MULTIPLY_INCLUDED__

/* 
    Backpropogation of the multiplication operation. Swithing case: where gradients are multiplied with the opposite 
    tensor. Example of a operation is as follows:
    
    F = x*y is forward propogation
    The gradients would be as follows:
    1. dF/dx = grad*y
    2. dF/dy = grad*x
*/
template <typename T>
void MultiplyOperation<T>::backward(Matrix<T> grad) {

    this->t1->backward(grad*this->t2->val);
    this->t2->backward(grad*this->t1->val);
}

/* 
    Forward Propogation of the operation. Returns a tensor.

    TODO: Remove: See addition operation impl for more details
*/
template <typename T>
Tensor<T> MultiplyOperation<T>::forwardDeprecated() {
    this->t3 = new Tensor<T>(this->t1->val * this->t2->val, this);
    return *this->t3;
}

/* 
    Forward Propogation of the operation. Return pointer to the tensor.
*/
template <typename T>
Tensor<T>* MultiplyOperation<T>::forward() {
    this->t3 = new Tensor<T>(this->t1->val * this->t2->val, this);
    return this->t3;
}

#endif

