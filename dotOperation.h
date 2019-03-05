#include "operation.h"

#ifndef __OP_DOT_INCLUDED__  
#define __OP_DOT_INCLUDED__  

template <typename T>
class DotOperation : public Operation<T> {
    public:
    Tensor<T> *t1 = NULL, *t2 = NULL; // Input tensors
    Tensor<T> *t3 = NULL; // Output tensor
    
    DotOperation(Tensor<T> *t1, Tensor<T> *t2) {
        this->t1 = t1;
        this->t2 = t2;
    }
    void backward(Matrix<T> grad);

    Tensor<T> forward();

};

#endif
