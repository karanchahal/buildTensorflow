/*
    This file defines the Stochastic Gradient Descent Optimiser
*/

#include "optims/optim.h"
#include "queue"

#ifndef __OPTIM_SGD_INCLUDED__   
#define __OPTIM_SGD_INCLUDED__ 


template<typename T>
class SGD : public Optimizer<T> {

    public: 

    SGD() {
        this->params.clear();
    }

    void getParams(Tensor<T>* x) {
        
        queue<Tensor<T>*> q;
        q.push(x);

        while(!q.empty()) {

            auto v = q.front();
            q.pop();
            auto op = v->backOp;

            if(op) {

                if(op->t1 != NULL && this->params.find(op->t1) == this->params.end()) {
                    q.push(op->t1);
                    this->params.insert(op->t1);
                }

                if(op->t2 != NULL && this->params.find(op->t2) == this->params.end()) {
                    q.push(op->t2);
                    this->params.insert(op->t2);
                }
            }
        }
    }

    void minimise(Tensor<T>* x, T lr) {

        // Get all tensors in computational grqaph
        getParams(x);

        // step through 1 parameter update
        step(lr);

        // reset Gradients to zero
        this->zeroGrad();
       
    }

    // Perform 1 step of learniung rate 
    void step(T learning_rate) {
        for(auto t: this->params) {
            t->val = t->val - learning_rate*t->grad;
        }
    }

};

#endif