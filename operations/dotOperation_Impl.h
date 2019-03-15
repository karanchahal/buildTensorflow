/*
This file contains the implementation of the forward and backward pass of
the dot operation.
*/

#include "operations/dotOperation.h"

#ifndef __OP_DOT_IMPL_INCLUDED__  
#define __OP_DOT_IMPL_INCLUDED__  

// TODO
template <typename T>
void DotOperation<T>::backward(Matrix<T> grad) {
    // TODO 
    
}

// TODO
template <typename T>
Tensor<T> DotOperation<T>::forward() {
    // TODO
    return *this->t1;
}

#endif

