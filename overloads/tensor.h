/*
    This file defines the various mathematical operations of the Tensor Class.
*/

#include <types/tensor.h>

#ifndef __TENSOR_OPS_INCLUDED__   
#define __TENSOR_OPS_INCLUDED__  

namespace tensorOps {

    // Addition 
    template<typename T>
    Tensor<T>* add(Tensor<T>* one, Tensor<T>* two) {
        one->frontOp = new AddOperation<T>(one, two);
        two->frontOp = one->frontOp;
        return one->frontOp->forward();
    }

    // Addition with Scalar - Scalar first
    template<typename T>
    Tensor<T>* add(T v, Tensor<T>* two) {
        auto one = new Tensor<T>(vector<T>(two->val.val.size(), v), two->val.shape);
        return add(one,two);
    }

    // Addition with Scalar - Vector first
    template<typename T>
    Tensor<T>* add(Tensor<T>* two, T v) {
        auto one = new Tensor<T>(vector<T>(two->val.val.size(), v), two->val.shape);
        return add(one,two);
    }

    // Subtraction
    template<typename T>
    Tensor<T>* subtract(Tensor<T>* one, Tensor<T>* two) {
        one->frontOp = new SubtractOperation<T>(one, two);
        two->frontOp = one->frontOp;
        return one->frontOp->forward();
    }

    // Subtraction with Scalar - Scalar first
    template<typename T>
    Tensor<T>* subtract(T v, Tensor<T>* two) {
        auto one = new Tensor<T>(vector<T>(two->val.val.size(),v),two->val.shape);
        return subtract(one,two);
    }

    // Subtraction with Scalar - Vector first
    template<typename T>
    Tensor<T>* subtract(Tensor<T>* two, T v) {
        auto one = new Tensor<T>(vector<T>(two->val.val.size(),v),two->val.shape);
        return subtract(one,two);
    }

    // Divide 
    template<typename T>
    Tensor<T>* divide(Tensor<T>* one, Tensor<T>* two) {
        one->frontOp = new DivideOperation<T>(one, two);
        two->frontOp = one->frontOp;
        return one->frontOp->forward();
    }

    // Divide Scalar - Scalar first
    template<typename T>
    Tensor<T>* divide(T v, Tensor<T>* two) {
        auto one = new Tensor<T>(vector<T>(two->val.val.size(), v), two->val.shape);
        return divide(one,two);
    }

    // Divide Scalar - Vector first
    template<typename T>
    Tensor<T>* divide(Tensor<T>* two, T v) {
        auto one = new Tensor<T>(vector<T>(two->val.val.size(), v), two->val.shape);
        return divide(one,two);
    }

    // Multiply
    template<typename T>
    Tensor<T>* multiply(Tensor<T>* one, Tensor<T>* two) {
        one->frontOp = new MultiplyOperation<T>(one, two);
        two->frontOp = one->frontOp;
        return one->frontOp->forward();
    }

    // Multiply with scalar - Scalar first
    template<typename T>
    Tensor<T>* multiply(T v, Tensor<T>* two) {
        auto one = new Tensor<T>(vector<T>(two->val.val.size(), v), two->val.shape);
        return multiply(one,two);
    }

    // Multiply with scalar - Vector first
    template<typename T>
    Tensor<T>* multiply(Tensor<T>* two, T v) {
        auto one = new Tensor<T>(vector<T>(two->val.val.size(), v), two->val.shape);
        return multiply(one,two);
    }

    // Dot Product
    template<typename T>
    Tensor<T>* dot(Tensor<T>* one, Tensor<T>* two) {
        one->frontOp = new DotOperation<T>(one, two);
        two->frontOp = one->frontOp;
        return one->frontOp->forward();
    }

    // Exponent 
    template<typename T>
    Tensor<T>* exp(Tensor<T>* one) {
        one->frontOp = new ExponentOperation<T>(one);
        return one->frontOp->forward();
    }

    // Sigmoid 
    template<typename T>
    Tensor<T>* sigmoid(Tensor<T>* one) {
        one->frontOp = new SigmoidOperation<T>(one);
        return one->frontOp->forward();
    }

    // Average
    template<typename T>
    Tensor<T>* average(vector<Tensor<T>*>& tensors) {
        if (tensors.size() == 0) {  // To avoid division by zero. Should we do this?
            return NULL;
        }

        Operation<T> op = new AverageOperation<T>(tensors);
        for (auto t : tensors) {
            t->frontOp = op;
        }

        return tensors[0]->frontOp->forward();
    }

};

#endif
