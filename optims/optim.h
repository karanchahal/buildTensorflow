/*
    This file defines the Base Class for all Optimizers.
*/

#include "types/tensor.h"
#include "unordered_set"

#ifndef __OPTIM_BASE_INCLUDED__   
#define __OPTIM_BASE_INCLUDED__ 

template<typename T>
class Optimizer {
    
    public:

    // This variable contains all the tensors that need to be updated via the optimiser
    unordered_set<Tensor<T>*> params;

    // The learning rate
    T lr;

    Optimizer() {
        
    }

    // This function resets the gradients of the tensors in params to zero for the next forward pass
    void zeroGrad() {
        for(auto i : params) {
            i->zeroGrad();
        }
    }

    // This overloaded function specifes how one optimisation step will be performed
    virtual void step(T learning_rate) {};
};

#endif