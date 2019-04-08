/*
    This file defines the Base Class for all Optimizers
*/

#include "types/tensor.h"
#include "set"

#ifndef __OPTIM_BASE_INCLUDED__   
#define __OPTIM_BASE_INCLUDED__ 

template<typename T>
class Optimizer {
    
    public:
    set<Tensor<T>*> params;

    Optimizer() {
        
    }

    void zeroGrad() {
        for(auto i : params) {
            i->zeroGrad();
        }
    }

    virtual void step(T learning_rate) {};
};

#endif