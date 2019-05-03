/*
    This file contains the implementation of the forward and backward pass of
    the add operation.
*/

#include "operations/add/addOperation.h"

#ifndef __OP_ADD_IMPL_INCLUDED__   
#define __OP_ADD_IMPL_INCLUDED__  

/* 
    Backpropogation of the addition operation. The addition operation distributes the gradient. So it
    effectively just transfers the gradient coming in to the various inpuot sources.
*/
template <typename T>
void AddOperation<T>::backward(Matrix<T> grad) {
    if(this->axis == -1) {
        
        if(this->t1->val.shape == this->t2->val.shape) {
            this->t1->backward(grad);
            this->t2->backward(grad);
        } else {

            //Example used for addition with biases
            // 64, 10 (1) added to 10 (2)
            // into 1 goes expanded 2 matrix
            // into 2 goes compressed 1 matrix
            
            this->t1->backward(grad);
            auto compressedGrad = grad.addAxis(0); 
            this->t2->backward(compressedGrad);
        }

    } else {
        int expansion = this->t1->val.shape[axis];
        auto expandedGrad = matrixOps::expandAlong(grad, axis, expansion);
        this->t1->backward(expandedGrad);
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
Tensor<T> AddOperation<T>::forwardDeprecated() {
    this->t3 = new Tensor<T>(this->t1->val + this->t2->val, this);
    return *this->t3;
}

/* 
    Forward Propogation of the operation. Return pointer to the tensor.
*/
template <typename T>
Tensor<T>* AddOperation<T>::forward() {
    if(axis == -1) {
        this->t3 = new Tensor<T>(this->t1->val + this->t2->val, this);
    } else {
        Matrix<T> res = this->t1->val.addAxis(axis);
        this->t3 = new Tensor<T>(res,this);
    }

    return this->t3;

}

#endif
