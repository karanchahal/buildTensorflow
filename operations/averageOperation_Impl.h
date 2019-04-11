/*
    This file contains the implementation of the forward and backward pass of
    the average operation.
*/

#include "operations/averageOperation.h"

#ifndef __OP_AVG_IMPL_INCLUDED__   
#define __OP_AVG_IMPL_INCLUDED__  

/* 
    Backpropogation of the average operation. The average operation distributes the gradient. So it
    effectively just transfers the gradient coming in to the various input sources after scaling
    it by the number of inputs.
*/
template <typename T>
void AverageOperation<T>::backward(Matrix<T> grad) {
    if (tensors.size() == 0) {  // To avoid division by zero
        return;
    }

    auto scaledGrad = grad / tensors.size();
    for(auto t : tensors) {
        t->backward(scaledGrad);
    }
}

/* 
    Forward Propogation of the average operation. Returns a tensor

    TODO: Remove: See average operation impl for more details
*/
template <typename T>
Tensor<T> AverageOperation<T>::forwardDeprecated() {
    return NULL;
}

/* 
    Forward Propogation of the operation. Return pointer to the tensor.
*/
template <typename T>
Tensor<T>* AverageOperation<T>::forward() {
    if (tensors.size() == 0) {   // To avoid division by zero
        return NULL;
    }

    Matrix<T> sum = NULL;

    for(auto t : tensors) {
        if(sum == NULL) {
            sum = t->val;
        }
        else {
            sum += t->val;
        }
    }

    sum /= tensors.size();

    this->t3 = new Tensor<T>(sum, this);
    return this->t3;
}

#endif
