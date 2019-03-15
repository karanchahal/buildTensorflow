/*
This file defines the Tensor class which stores the matrices used for
computations by the library, as well as the forward and backward operations
being executed on it in the form of pointers to Operation objects.
*/

#include "operations/operation.h"
#include "operations/addOperation.h"
#include "operations/multiplyOperation.h"
#include "operations/divideOperation.h"
#include "operations/exponentOperation.h"
#include "operations/dotOperation.h"

#ifndef __TENSOR_FLOAT_INCLUDED__   
#define __TENSOR_FLOAT_INCLUDED__   

template <typename T>
class Tensor {
    public:
    Matrix<T> val; // value of tensor 
    Matrix<T> grad; // value of grad

    Operation<T> *frontOp =NULL, *backOp =NULL;

    Tensor() {
        // default
        cout<<"Normal Constructor is called"<<endl;
    }

    /*
        This is a copy constructor needed for when we encounter temporaries.
        In a big expression, for example a*(b+c) has the (b+c) temporary
        which comes in the form of const Tensor, this const Tensor gives problems and
        weird values for grad and val when backpropping through it.
        We create a normal non const Tensor whenever we encounter this const Tensor, 
        hence this copy constructor is used. Check the const operator overloading for
        usage of this constructor.
    */
    Tensor(const Tensor<T> *two) {
        this->val = two->val;
        this->backOp = two->backOp;
        this->frontOp = two->frontOp;
        this->grad = two->grad;
    }

    Tensor(Matrix<T> &val) {
        this->val = val;
    }

    Tensor(vector<T> val, vector<int> shape) {
        this->val = Matrix<T>(val,shape);
    }

    Tensor(Matrix<T> val, Operation<T>* op) {
        this->val = val;
        this->backOp = op;
    }

    void backward(Matrix<T> grad) {
        // TODO: add assertions that the gradient is of the same shape of val
        this->grad = grad;
        if(this->backOp != NULL) {
            this->backOp->backward(grad);
        }
    }

    // Overloading Operations, one for const and one for actual variable
    // hence each operation has two overloads, except exp()
    Tensor<T> operator * (Tensor<T> &two) { 
        // TODO: add assertions
        this->frontOp = new MultiplyOperation<T>(this, &two);
        two.frontOp = this->frontOp;
        return this->frontOp->forward();
    }

    Tensor<T> operator * (const Tensor<T> &two) { 
        // TODO: add assertions
        Tensor<T>* temp = new Tensor<T>(&two);
        return (*this)*(*temp);
    }

    Tensor<T> operator + (Tensor<T> &two) { 
        // TODO: add assertions
        this->frontOp = new AddOperation<T>(this, &two);
        two.frontOp = this->frontOp;
        return this->frontOp->forward();
    }

    Tensor<T> operator + (const Tensor<T> &two) { 
        // TODO: add assertions
        Tensor<T>* temp = new Tensor<T>(&two);
        return (*this)+(*temp);
    }

    Tensor<T> operator / (Tensor<T> &two) { 
        // TODO: add assertions
        this->frontOp = new DivideOperation<T>(this, &two);
        two.frontOp = this->frontOp;
        return this->frontOp->forward();
    }

    Tensor<T> operator / (const Tensor<T> &two) { 
        // TODO: add assertions
        Tensor<T>* temp = new Tensor<T>(&two);
        return (*this)/(*temp);
    }

    // Dot Product
    Tensor<T> dot(Tensor<T> &two) { 
        // TODO: add assertions
        this->frontOp = new DotOperation<T>(this, &two);
        two.frontOp = this->frontOp;
        return this->frontOp->forward();
    }

    Tensor<T> exp() { 
        // TODO: add assertions
        this->frontOp = new ExponentOperation<T>(this);
        return this->frontOp->forward();
    }

};

#endif

