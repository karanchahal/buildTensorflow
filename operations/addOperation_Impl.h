/*
This file contains the implementation of the forward and backward pass of
the add operation.
*/

#include "operations/addOperation.h"

#ifndef __OP_ADD_IMPL_INCLUDED__   
#define __OP_ADD_IMPL_INCLUDED__  

template <typename T>
void AddOperation<T>::backward(Matrix<T> grad) {
    // Distributing case: where gradients are backproped as is
    this->t1->backward(grad);
    this->t2->backward(grad);
}

template <typename T>
Tensor<T> AddOperation<T>::forward() {
    this->t3 = new Tensor<T>(this->t1->val + this->t2->val, this);
    return *this->t3;
}

#endif
