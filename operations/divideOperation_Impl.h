/*
This file contains the implementation of the forward and backward pass of
the divide operation.
*/

#include "operations/divideOperation.h"

#ifndef __OP_DIV_IMPL_INCLUDED__   
#define __OP_DIV_IMPL_INCLUDED__  

// x/y is forward
// for dF/dx = grad/y
// dF/dy = (grad*x* (-1))/y^2
template <typename T>
void DivideOperation<T>::backward(Matrix<T> grad) {
    // Swithing case: weherre gradients are multiplied with the opposite tensor
    this->t1->backward(grad / this->t2->val);
    auto temp = grad * this->t1->val;
    this->t2->backward(temp*( (T)(-1) / (this->t2->val^(T)2)));
}

template <typename T>
Tensor<T> DivideOperation<T>::forward() {
    this->t3 = new Tensor<T>(this->t1->val / this->t2->val, this);
    return *this->t3;
}

#endif

