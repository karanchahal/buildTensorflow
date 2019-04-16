/*
    This file contains the implementation of the forward and backward pass of
    the exponent operation.
*/

#include "operations/log/logOperation.h"

#ifndef __OP_LOG_IMPL_INCLUDED__   
#define __OP_LOG_IMPL_INCLUDED__   

/* 
    Backpropogation of the log operation. Example of a operation is as follows:
    F = log(x) is forward propogation
    The gradients would be as follows:
    1. dF/dx = grad/x
*/
template <typename T>
void LogOperation<T>::backward(Matrix<T> grad) {
    this->t1->backward(grad/this->t1->val);
}

/* 
    Forward Propogation of the operation. Returns a tensor.

    TODO: Remove: See addition operation impl for more details
*/
template <typename T>
Tensor<T> LogOperation<T>::forwardDeprecated() {
    return NULL;
}

/* 
    Forward Propogation of the operation. Return pointer to the tensor.
*/
template <typename T>
Tensor<T>* LogOperation<T>::forward() {
    this->t3 = new Tensor<T>(this->t1->val.log(), this);
    return this->t3;
}

#endif

