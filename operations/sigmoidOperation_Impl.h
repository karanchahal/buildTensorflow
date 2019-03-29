/*
This file contains the implementation of the forward and backward pass of
the sigmoid operation.
*/

#include "operations/sigmoidOperation.h"

#ifndef __OP_IMPL_SIG_INCLUDED__
#define __OP_IMPL_SIG_INCLUDED__

template <typename T>
void SigmoidOperation<T>::backward(Matrix<T> grad) {
    // Switching case: where gradients are multiplied with the opposite tensor
    // TODO refactor matrix overloads and add scalar/matrix operations
    auto one = Matrix<T>(vector<T>(this->t3->val.val.size(),1),this->t3->val.shape);
    auto minusone = Matrix<T>(vector<T>(this->t3->val.val.size(),-1),this->t3->val.shape);
    
    auto g = (one + minusone*this->t3->val)*this->t3->val; // matrix can be overloaded
    this->t1->backward(grad*g);
}

template <typename T>
Tensor<T> SigmoidOperation<T>::forward() {
   // Not implemented as yet
   return NULL;
}

template <typename T>
Tensor<T>* SigmoidOperation<T>::forwardPointer() {
    // TODO
    // Make function that does this for matrices using overloads
    auto d = tensorOps::multiply((T)-1,this->t1);
    auto e = tensorOps::exp(d);
    auto f = tensorOps::add((T)1,e);
    auto g = tensorOps::divide((T)1,f);

    return g;
}

#endif

