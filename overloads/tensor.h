/*
    This file defines the various mathematical operations of the Tensor Class.
*/

#include <types/tensor.h>

#include "operations/add/addOperation.h"
#include "operations/multiply/multiplyOperation.h"
#include "operations/divide/divideOperation.h"
#include "operations/exponent/exponentOperation.h"
#include "operations/log/logOperation.h"
#include "operations/dot/dotOperation.h"
#include "operations/sigmoid/sigmoidOperation.h"
#include "operations/power/powerOperation.h"
#include "operations/softmax/softmaxoperation.h"
#include "operations/average/averageOperation.h"


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

    // Addition with Scalar
    template<typename T>
    Tensor<T>* add(T v, Tensor<T>* two) {
        auto one = new Tensor<T>(vector<T>(two->val.val.size(), v), two->val.shape);
        return add(one,two);
    }

    // Addition with Scalar
    template<typename T>
    Tensor<T>* add(Tensor<T>* one, int axis) {
        one->frontOp = new AddOperation<T>(one, axis);
        return one->frontOp->forward();
    }

    // Divide 
    template<typename T>
    Tensor<T>* divide(Tensor<T>* one, Tensor<T>* two) {
        one->frontOp = new DivideOperation<T>(one, two);
        two->frontOp = one->frontOp;
        return one->frontOp->forward();
    }

    // Divide Scalar
    template<typename T>
    Tensor<T>* divide(T v, Tensor<T>* two) {
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

    // Multiply with scalar
    template<typename T>
    Tensor<T>* multiply(T v, Tensor<T>* two) {
        auto one = new Tensor<T>(vector<T>(two->val.val.size(), v), two->val.shape);
        one->requires_grad=false;
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

    // Log 
    template<typename T>
    Tensor<T>* log(Tensor<T>* one) {
        one->frontOp = new LogOperation<T>(one);
        return one->frontOp->forward();
    }


    // Sigmoid 
    template<typename T>
    Tensor<T>* sigmoid(Tensor<T>* one) {
        one->frontOp = new SigmoidOperation<T>(one);
        return one->frontOp->forward();
    }

    // Power
    template<typename T>
    Tensor<T>* power(Tensor<T>* one, T t) {
        one->frontOp = new PowerOperation<T>(one, t);
        return one->frontOp->forward();
    }

    // Softmax
    template<typename T>
    Tensor<T>* softmax(Tensor<T>* one, int axis) {
        one->frontOp = new SoftmaxOperation<T>(one, axis);
        return one->frontOp->forward();
    }
    // Average
    template<typename T>
    Tensor<T>* average(Tensor<T>* one, int axis) {
        one->frontOp = new AverageOperation<T>(one, axis);
        return one->frontOp->forward();
    }


};

#endif
