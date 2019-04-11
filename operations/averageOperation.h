/*
    This file defines the AverageOperation class which represents the average of
    multiple tensors.
*/

#include "operations/operation.h"

#ifndef __OP_AVG_INCLUDED__   
#define __OP_AVG_INCLUDED__   

template <typename T>
class AverageOperation : public Operation<T> {
    public:

    vector<Tensor<T*>> tensors;
    
    AverageOperation(vector<Tensor<T>*>& tensors) {
        this->tensors = tensors;
    }

    void backward(Matrix<T> grad);

    Tensor<T> forwardDeprecated();

    Tensor<T>* forward();

};

#endif
