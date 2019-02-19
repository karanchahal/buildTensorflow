#include "multiplyOperation.h"
#include "addOperation.h"
#include "divideOperation.h"
#include "tensor.h"
#include "overloads.h"

template <typename T>
void MultiplyOperation<T>::backward(vector<T> grad) {
    // cout<<this->t2->val<<endl;
    // Swithing case: weherre gradients are multiplied with the opposite tensor
    this->t1->backward(grad*this->t2->val);
    this->t2->backward(grad*this->t1->val);
}

template <typename T>
Tensor<T> MultiplyOperation<T>::forward() {
    return Tensor<T>(this->t1->val*this->t2->val, this);
}

template <typename T>
void AddOperation<T>::backward(vector<T> grad) {
    // cout<<this->t2->val<<endl;
    // Distributing case: weherre gradients are backproped as is
    this->t1->backward(grad);
    this->t2->backward(grad);
}

template <typename T>
Tensor<T> AddOperation<T>::forward() {
    return Tensor<T>(this->t1->val+this->t2->val, this);
}

// x/y is forward
// for dF/dx = grad/y
// dF/dy = (grad*x* (-1))/y^2
template <typename T>
void DivideOperation<T>::backward(vector<T> grad) {
    // cout<<this->t2->val<<endl;
    // Swithing case: weherre gradients are multiplied with the opposite tensor
    this->t1->backward(grad/this->t2->val);
    vector<T> temp = grad*this->t1->val;
    this->t2->backward(temp*( (T)(-1)/ (this->t2->val^(T)2)));
}

template <typename T>
Tensor<T> DivideOperation<T>::forward() {
    return Tensor<T>(this->t1->val*this->t2->val, this);
}

// Exponent Backprop
template <typename T>
void ExponentOperation<T>::backward(vector<T> grad) {
    this->t1->backward(grad*exponent(this->t1->val));
}

template <typename T>
Tensor<T> ExponentOperation<T>::forward() {
    return Tensor<T>(exponent(this->t1->val), this);
}




