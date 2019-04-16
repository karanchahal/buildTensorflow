/*
    This file defines the AddOperation class which represents the addition of
    two tensors.
*/

#include "operations/operation.h"

#ifndef __OP_AVG_INCLUDED__   
#define __OP_AVG_INCLUDED__   

template <typename T>
class AverageOperation : public Operation<T> {
    public:
    
    int axis = -1;

    AverageOperation(Tensor<T> *t1, int axis) {
        this->t1 = t1;
        this->axis = axis;
    }

    void backward(Matrix<T> grad);

    Tensor<T> forwardDeprecated();

    Tensor<T>* forward();

};

#endif

