/*
    This file contains the implementation of the forward and backward pass of
    the subtract operation.
*/

#include "operations/subtractOperation.h"

#ifndef __OP_SUBTRACT_IMPL_INCLUDED__   
#define __OP_SUBTRACT_IMPL_INCLUDED__  

template <typename T>
void SubtractOperation<T>::backward(Matrix<T> grad) {
    // Distributing case with negative: where one gradients is backproped
    // as is, and the other is backproped with a negative sign
    this->t1->backward(grad);
    this->t2->backward(-1 * grad);
}

template <typename T>
Tensor<T>* SubtractOperation<T>::forward() {
    this->t3 = new Tensor<T>(this->t1->val - this->t2->val, this);
    return this->t3;
}

template <typename T>
Tensor<T> SubtractOperation<T>::forwardDeprecated() {
    this->t3 = new Tensor<T>(this->t1->val - this->t2->val, this);
    return *this->t3;
}

#endif
