/*
 This file defines the various mathematical overloads of the Tensor Class
*/
#include <types/tensor.h>

namespace tensorOps {

    template<typename T>
    Tensor<T>* add(Tensor<T>* one, Tensor<T>* two) {
        one->frontOp = new AddOperation<T>(one, two);
        two->frontOp = one->frontOp;
        return one->frontOp->forwardPointer();
    }

    template<typename T>
    Tensor<T>* divide(Tensor<T>* one, Tensor<T>* two) {
        one->frontOp = new DivideOperation<T>(one, two);
        two->frontOp = one->frontOp;
        return one->frontOp->forwardPointer();
    }

    template<typename T>
    Tensor<T>* multiply(Tensor<T>* one, Tensor<T>* two) {
        one->frontOp = new MultiplyOperation<T>(one, two);
        two->frontOp = one->frontOp;
        return one->frontOp->forwardPointer();
    }

    template<typename T>
    Tensor<T>* dot(Tensor<T>* one, Tensor<T>* two) {
        one->frontOp = new DotOperation<T>(one, two);
        two->frontOp = one->frontOp;
        return one->frontOp->forwardPointer();
    }


    template<typename T>
    Tensor<T>* exp(Tensor<T>* one) {
        one->frontOp = new ExponentOperation<T>(one);
        return one->frontOp->forwardPointer();
    }

}
