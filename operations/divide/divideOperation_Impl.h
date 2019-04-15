/*
This file contains the implementation of the forward and backward pass of
the divide operation.
*/

#include "operations/divide/divideOperation.h"

#ifndef __OP_DIV_IMPL_INCLUDED__   
#define __OP_DIV_IMPL_INCLUDED__  

/* 
    Backpropogation of the division operation. Example of a division operation is as follows:
    F = x/y is forward propogation
    The gradients would be as follows:
    1. dF/dx = grad/y
    2. dF/dy = (grad*x* (-1))/y^2
*/
template <typename T>
void DivideOperation<T>::backward(Matrix<T> grad) {
    
    this->t1->backward(grad / this->t2->val);
    auto temp = ((T)(-1))*grad*this->t1->val;
    this->t2->backward(temp/(this->t2->val^((T)2)));
}

/* 
    Forward Propogation of the operation. Returns a tensor.

    TODO: Remove: See addition operation impl for more details
*/
template <typename T>
Tensor<T> DivideOperation<T>::forwardDeprecated() {
    this->t3 = new Tensor<T>(this->t1->val / this->t2->val, this);
    return *this->t3;
}

/* 
    Forward Propogation of the operation. Return pointer to the tensor.
*/
template <typename T>
Tensor<T>* DivideOperation<T>::forward() {
    this->t3 = new Tensor<T>(this->t1->val / this->t2->val, this);
    return this->t3;
}

#endif

