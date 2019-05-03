/*
    This file contains the implementation of the forward and backward pass of
    the add operation.
*/

#include "operations/average/averageOperation.h"

#ifndef __OP_AVG_IMPL_INCLUDED__   
#define __OP_AVG_IMPL_INCLUDED__  

/* 
    Backpropogation of the addition operation. The addition operation distributes the gradient. So it
    effectively just transfers the gradient coming in to the various inpuot sources.
*/
template <typename T>
void AverageOperation<T>::backward(Matrix<T> grad) {
  
    int expansion = this->t1->val.shape[axis];
    if(this->t1->val.shape == grad.shape) {
        this->t1->backward(grad);
    } else {
        auto expandedGrad = matrixOps::expandAlong(grad, axis, expansion);
        // Temporary Fix: TODO make consistent across codebase
        // Removing extra dimension from shape
        expandedGrad.squeeze(1);
        this->t1->backward(expandedGrad/(T)expansion);
    }

}

/* 
    Forward Propogation of the addition operation. Returns a tensor

    TODO: Deprecated API, need to figure this out. As it returns a tensor. It becomes difficult to track pointers
    in a computational graph. Pointers should be used everywhere to 
    1. Save memory
    2. Avoid weird allocation issues
*/
template <typename T>
Tensor<T> AverageOperation<T>::forwardDeprecated() {
    this->t3 = new Tensor<T>(this->t1->val + this->t2->val, this);
    return *this->t3;
}

/* 
    Forward Propogation of the operation. Return pointer to the tensor.
*/
template <typename T>
Tensor<T>* AverageOperation<T>::forward() {
   
    Matrix<T> res = this->t1->val.addAxis(axis) / (T)this->t1->val.shape[axis];
    this->t3 = new Tensor<T>(res,this);

    return this->t3;

}

#endif
