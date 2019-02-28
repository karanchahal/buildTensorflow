#include "operation.h"


#ifndef __OP_ADD_INCLUDED__   
#define __OP_ADD_INCLUDED__   

template <typename T>
class AddOperation : public Operation<T> {

    public:
        Tensor<T> *t1, *t2; // generally an operation had two operands
        Tensor<T> *t3;
    
    AddOperation(Tensor<T> *t1, Tensor<T> *t2) {
        this->t1 = t1;
        this->t2 = t2;
    }
    void backward(vector<T> grad);

    Tensor<T> forward();
};
#endif
