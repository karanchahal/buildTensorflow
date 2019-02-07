#include "operation.h"


#ifndef __OP_ADD_INCLUDED__   
#define __OP_ADD_INCLUDED__   

class AddOperation : public Operation {

    public:
        FloatTensor *t1, *t2; // generally an operation had two operands
    
    AddOperation(FloatTensor *t1, FloatTensor *t2) {
        this->t1 = t1;
        this->t2 = t2;
    }
    void backward(float grad);

    FloatTensor compute();
};
#endif
