#ifndef __OP_H_INCLUDED__   
#define __OP_H_INCLUDED__   

#include<iostream>
#include<vector>

using namespace std; 

template<typename T>
class Tensor;

template<typename T>
class Operation {
    public:
        Tensor<T> *t1, *t2; // generally an operation had two operands

        virtual void backward(vector<T> grad) = 0;
        virtual Tensor<T> forward() = 0;
};

#endif