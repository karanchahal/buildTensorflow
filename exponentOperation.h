/*
This file defines the ExponentOperation class which represents the
exponentiation of a tensor.
*/

#include "operation.h"

#ifndef __OP_EXP_INCLUDED__   
#define __OP_EXP_INCLUDED__   

template <typename T>
class ExponentOperation : public Operation<T> {
    public:
        Tensor<T> *t1; // Input tensor
        Tensor<T> *t3; // Output tensor
    
    ExponentOperation(Tensor<T> *t1) {
        this->t1 = t1;
    }

    void backward(vector<T> grad);
    Tensor<T> forward();
};
#endif
