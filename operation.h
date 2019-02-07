#ifndef __OP_H_INCLUDED__   
#define __OP_H_INCLUDED__   

#include<iostream>

using namespace std; 

class FloatTensor;

class Operation {
    public:
        FloatTensor *t1, *t2; // generally an operation had two operands

        virtual void backward(float grad) = 0;
        virtual FloatTensor compute() = 0;
};

#endif