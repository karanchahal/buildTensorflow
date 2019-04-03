/*
This file contains the implementation of the forward and backward pass of
the sigmoid operation.
*/

#include "operations/reluOperation.h"

#ifndef __OP_IMPL_RELU_INCLUDED__
#define __OP_IMPL_RELU_INCLUDED__

template <typename T>
void ReluOperation<T>::backward(Matrix<T> grad) {
    int n = this->t1->val.val.size();

    for(int i = 0; i < n;i++) {
        if(this->t1->val.val[i] > 0) {
            grad.val[i] = 0;
        }
    }

    return this->t1->backward(grad);
}

template <typename T>
Tensor<T> ReluOperation<T>::forward() {
   // Not implemented as yet
   return NULL;
}

template <typename T>
Tensor<T>* ReluOperation<T>::forwardPointer() {
    // TODO
    // Make function that does this for matrices using overloads
    int n = this->t1->val.val.size();
    vector<T> v(n,0);
    for(int i = 0; i < n;i++) {
        if(this->t1->val.val[i] > 0) {
            v[i] = this->t1->val.val[i];
        }
    }
    this->t3 = new Tensor<T>(v, this->t1->shape);
    return this->t3;
}

#endif

