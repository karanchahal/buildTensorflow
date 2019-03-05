/*
This file contains the implementation of the forward and backward pass of
the operations.
*/

#include "multiplyOperation.h"
#include "addOperation.h"
#include "divideOperation.h"
#include "dotOperation.h"

// We want to define forward and backprop relating to both vector to vector, vector to matrix and matrix to matrix operations.
// multiplication is dot product multiplication
// addition is element wise addition
// division is elementwise multiplication

template <typename T>
void MultiplyOperation<T>::backward(Matrix<T> grad) {
    // Swithing case: where gradients are multiplied with the opposite tensor
    this->t1->backward(grad*this->t2->val);
    this->t2->backward(grad*this->t1->val);
}

template <typename T>
Tensor<T> MultiplyOperation<T>::forward() {
    this->t3 = new Tensor<T>(this->t1->val * this->t2->val, this);
    return *this->t3;
}

// TODO
template <typename T>
void DotOperation<T>::backward(Matrix<T> grad) {
    // TODO 
    
}

// TODO
template <typename T>
Tensor<T> DotOperation<T>::forward() {
    // TODO
    return *this->t1;
}

template <typename T>
void AddOperation<T>::backward(Matrix<T> grad) {
    // Distributing case: where gradients are backproped as is
    this->t1->backward(grad);
    this->t2->backward(grad);
}

template <typename T>
Tensor<T> AddOperation<T>::forward() {
    this->t3 = new Tensor<T>(this->t1->val + this->t2->val, this);
    return *this->t3;
}

// x/y is forward
// for dF/dx = grad/y
// dF/dy = (grad*x* (-1))/y^2
template <typename T>
void DivideOperation<T>::backward(Matrix<T> grad) {
    // Swithing case: weherre gradients are multiplied with the opposite tensor
    this->t1->backward(grad / this->t2->val);
    auto temp = grad * this->t1->val;
    this->t2->backward(temp*( (T)(-1) / (this->t2->val^(T)2)));
}

template <typename T>
Tensor<T> DivideOperation<T>::forward() {
    this->t3 = new Tensor<T>(this->t1->val / this->t2->val, this);
    return *this->t3;
}

// Exponent Backprop
template <typename T>
void ExponentOperation<T>::backward(Matrix<T> grad) {
    this->t1->backward(grad*(this->t1->val.exp()));
}

template <typename T>
Tensor<T> ExponentOperation<T>::forward() {
    this->t3 = new Tensor<T>(this->t1->val.exp(), this);
    return *this->t3;
}
