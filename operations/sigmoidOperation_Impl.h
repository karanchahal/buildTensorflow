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
    auto g = (1 - t3->val)*t3->val;
    this->t1->backward(grad*g);
}

template <typename T>
Tensor<T> SigmoidOperation<T>::forward() {
   // TODO
}

template <typename T>
Tensor<T>* SigmoidOperation<T>::forwardPointer() {
    // TODO
    // Make function that does this for matrices using overloads
}

#endif

