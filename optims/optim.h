/*
    This file defines the Base Class for all Optimizers
*/

#include "types/tensor.h"
#include "unordered_set"

#ifndef __OPTIM_BASE_INCLUDED__   
#define __OPTIM_BASE_INCLUDED__ 

template<typename T>
class Optimizer {
    
    public:
    unordered_set<Tensor<T>*> params;
    T lr;

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