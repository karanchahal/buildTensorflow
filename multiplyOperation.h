#include "operation.h"

#ifndef __OP_MULTIPLY_INCLUDED__   // if operation.h hasn't been included yet...
#define __OP_MULTIPLY_INCLUDED__   //   #define this so the compiler knows it has been included

class MultiplyOperation : public Operation {

    public:
        FloatTensor *t1, *t2; // generally an operation had two operands
    
    MultiplyOperation(FloatTensor *t1, FloatTensor *t2) {
        this->t1 = t1;
        this->t2 = t2;
    }
    void backward(float grad);

    FloatTensor compute();
};

#endif