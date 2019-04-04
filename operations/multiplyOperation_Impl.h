/*
This file contains the implementation of the forward and backward pass of
the multiply operation.
*/

#include "operations/multiplyOperation.h"

#ifndef __OP_IMPL_MULTIPLY_INCLUDED__
#define __OP_IMPL_MULTIPLY_INCLUDED__

template <typename T>
void MultiplyOperation<T>::backward(Matrix<T> grad) {
    // Swithing case: where gradients are multiplied with the opposite tensor
    this->t1->backward(grad*this->t2->val);
    this->t2->backward(grad*this->t1->val);
}

template <typename T>
Tensor<T> MultiplyOperation<T>::forward() {
    this->t3 = new Tensor<T>(this->t1->val * this->t2->val, this);
    return *this->t3;
}

#endif

