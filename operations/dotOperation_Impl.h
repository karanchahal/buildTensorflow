/*
This file contains the implementation of the forward and backward pass of
the dot operation.
*/

#include "operations/dotOperation.h"
#include "utils/matrix.h"

#ifndef __OP_DOT_IMPL_INCLUDED__  
#define __OP_DOT_IMPL_INCLUDED__  

// TODO: Add comments
template <typename T>
void DotOperation<T>::backward(Matrix<T> grad) {
    // TODO: Cleanup code for this
    vector<int> trsIndx(this->t1->val.shape.size());
    for(int i = 0;i<trsIndx.size()-2;i++) {
        trsIndx[i] = i;
    }
    trsIndx[trsIndx.size()-1] = trsIndx.size() - 2;
    trsIndx[trsIndx.size()-2] = trsIndx.size() - 1;

    auto one = Matrix<T>(grad.dot(utils::transpose<T>(this->t2->val,{1,0})));
    auto two = Matrix<T>(utils::transpose<T>(this->t1->val,trsIndx).dot(grad));
    this->t1->backward(one);
    this->t2->backward(two);
}

// TODO: Add comments
template <typename T>
Tensor<T> DotOperation<T>::forward() {
    // TODO: Add comments
    auto val3 = this->t1->val.dot(this->t2->val);
    this->t3 = new Tensor<T>(val3,this);
    return *this->t3;
}

#endif

