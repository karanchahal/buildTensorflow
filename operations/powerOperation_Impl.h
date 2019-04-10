/*
    This file contains the implementation of the forward and backward pass of
    the power operation.
*/

#include "operations/powerOperation.h"

#ifndef __OP_IMPL_POWER_INCLUDED__
#define __OP_IMPL_POWER_INCLUDED__

/* 
    Backpropogation of the power operation.
    
    F = x*pow is forward propogation
    The gradient would be as follows:
    1. dF/dx = pow*x^(pow-1)
*/
template <typename T>
void PowerOperation<T>::backward(Matrix<T> grad) {
    this->t1->backward(grad * (pow * matrixOps::power(this->t1->val,pow-1)));
}

/* 
    Forward Propogation of the operation. Returns a tensor.

    TODO: Remove: See addition operation impl for more details
*/
template <typename T>
Tensor<T> PowerOperation<T>::forwardDeprecated() {
    return NULL;
}

/* 
    Forward Propogation of the operation. Return pointer to the tensor.
    Forward propogation is simply y = x^(pow).
*/
template <typename T>
Tensor<T>* PowerOperation<T>::forward() {
    this->t3 = new Tensor<T>(matrixOps::power(this->t1->val, this->pow), this);
    return this->t3;
}

#endif

