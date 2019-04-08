/*
    This file defines the Tensor class. The Tensor is the base object used for creating 
    neural networks. Each neuron in the network will be a tensor. In this project we will 
    use matrices to represent layers of a neural network, hence each tensor would simply be
    a n dimensional matrix. 
    
    Apart from this the tensor is aware of what operations have been performed with it. This 
    is useful when we want to calculate the gradients of the tensor with respect to some
    other variable (a loss function generally with neural networks).
    To calculate these gradients, the tensor keeps a track of what operations have been 
    performed with it as a operand. 
    
    For now we are assuming that the tensor just goes through one operation, hence we have 
    a frontOp denoting the operation that it is involved in and backOp is the operation that 
    created this tensor.

    To know more about Operations head to the operation module and check out the operation.h
    file.
*/

#include "operations/operation.h"
#include "operations/addOperation.h"
#include "operations/multiplyOperation.h"
#include "operations/divideOperation.h"
#include "operations/exponentOperation.h"
#include "operations/dotOperation.h"
#include "operations/sigmoidOperation.h"
#include "utils/matrix.h"

#ifndef __TENSOR_FLOAT_INCLUDED__   
#define __TENSOR_FLOAT_INCLUDED__   

template <typename T>
class Tensor {

    public:
    
    /*
        Value of the Tensor: Head to type/matrix.h to know more about Matrices which
        store n dimensional arrays.
    */
    Matrix<T> val; 

    /*
        Gradient value of the tensor, also stored as a Matrix.
        TODO: Initialise to zero and keep it as same shape as val 
    */
    Matrix<T> grad; 

    /*
        The Operations that the tensor is related with. The frontOp denotes the operation
        that the tensor is an operand to. The backOp operation is the operation which led
        to the creation of this tensor.
    */
    Operation<T> *frontOp =NULL, *backOp =NULL;

    /*
        The default constructor of the Tensor. This function is not really used anywhere in
        the program as creating an empty Tensor is essentially meaningless.
        
        TODO: Check if we really need a default constructor
    */
    Tensor() {
    }

    /*
        This is a copy constructor needed for when we need to copy a tensor out to a new
        tensor object.
    */
    Tensor(const Tensor<T> *two) {
        this->val = two->val;
        this->backOp = two->backOp;
        this->frontOp = two->frontOp;
        this->grad = two->grad;
    }

    // Constructor to create Tensor from a Matrix
    Tensor(Matrix<T> &val) {
        this->val = val;
        zeroGrad();
    }

    /*
        Constructor to create Tensor from a value vector and shape vector.
        The matrix class takes as input these two parameters and this constructor provides
        an easy way to create tensors.
    */
    Tensor(vector<T> val, vector<int> shape) {
        this->val = Matrix<T>(val, shape);
        zeroGrad();
    }

    /*
        The constructor is used by an Operation when it needs to create a new tensor 
        which is the result of that operation with it's operands. In such a scenario we
        store a reference of the operation that created said tensor so that we can store
        it in the backOp variable for when we need to do backward Propogation of our
        computational graph.
    */
    Tensor(Matrix<T> val, Operation<T>* op) {
        this->val = val;
        this->backOp = op;
        zeroGrad();
    }

    /*
        Entry Function from where backward Propogation is performed to train our network.
        Or shall we say, where the magic happens ! The backward propogation operates in a
        depth first serach manner going through all the nodes in the computational graph
        carrying the gradient w.r.t the tensor around with it. Backprop ends when we are at
        the first layer of our network, i.e when we have no more backward Operations to 
        traverse.

        TODO: This is a DFS style backward Call. Write robust tests to validate backward call.
    */
    void backward(Matrix<T> grad) {
        assert(grad.shape == val.shape && "The gradient and the tensor shapes do not match !");
        this->grad = this->grad + grad;
        if(this->backOp != NULL) {
            this->backOp->backward(grad);
        }
    }

    void backward() {
        // Make gradient of all 1's
        vector<T> v(val.val.size(),1);
        auto grad = Matrix<T>(v, val.shape);
        
        this->grad = this->grad + grad;
        if(this->backOp != NULL) {
            this->backOp->backward(grad);
        }
    }

    /*
        This function is called during the initilaisation of Tensor. It sets the value of it's gradients to zero. This is needed as 
        during backPropogation the same tensor can be used for different operation, hence to calculate it's partial gradients
        each individual operation's gradients have to be summed up. Hence we initialise the tensor's gradients to zero.
        
        See constructor for it's usage.
    */
    void zeroGrad() {
        assert(val.shape.size() != 0 && "The value of matrix cannot be uninitialised during initialisng zeros in tensor's gradient");
        vector<T> g(val.val.size(), 0);
        this->grad = Matrix<T>(g, val.shape);
    }

    /*
        From here on, we overload the operators like +, / and * to define what happens when
        we we add, divide and multiply tensors. We also support other operations like dot 
        product (used heavily in fully connected, convolution and recurrent networks).
        We also support some operations for our actiavtion functions like exponent, power and
        maybe more.
    */

    /*
        Elementwise Multiplication between 2 tensors
    */
    Tensor<T> operator * (Tensor<T> &two) { 
        // TODO: add assertions
        this->frontOp = new MultiplyOperation<T>(this, &two);
        two.frontOp = this->frontOp;
        return this->frontOp->forwardDeprecated();
    }

    Tensor<T> operator * (const Tensor<T> &two) { 
        // TODO: add assertions
        Tensor<T>* temp = new Tensor<T>(&two);
        return (*this)*(*temp);
    }

    /*
        Elementwise Addition between 2 tensors
    */
    Tensor<T> operator + (Tensor<T> &two) { 
        // TODO: add assertions
        this->frontOp = new AddOperation<T>(this, &two);
        two.frontOp = this->frontOp;
        return this->frontOp->forwardDeprecated();
    }

    Tensor<T> operator + (const Tensor<T> &two) { 
        // TODO: add assertions
        Tensor<T>* temp = new Tensor<T>(&two);
        return (*this)+(*temp);
    }

    /*
        Elementwise Division between 2 tensors
    */
    Tensor<T> operator / (Tensor<T> &two) { 
        // TODO: add assertions
        this->frontOp = new DivideOperation<T>(this, &two);
        two.frontOp = this->frontOp;
        return this->frontOp->forwardDeprecated();
    }

    Tensor<T> operator / (const Tensor<T> &two) { 
        // TODO: add assertions
        Tensor<T>* temp = new Tensor<T>(&two);
        return (*this)/(*temp);
    }

    /*
        Matrix Multiplication or Dot Product between 2 tensors
    */
    Tensor<T> dot(Tensor<T> &two) { 
        // TODO: add assertions
        this->frontOp = new DotOperation<T>(this, &two);
        two.frontOp = this->frontOp;
        return this->frontOp->forwardDeprecated();
    }

    /*
        Elementwise Exponent Calculation of tensor
    */
    Tensor<T> exp() { 
        // TODO: add assertions
        this->frontOp = new ExponentOperation<T>(this);
        return this->frontOp->forwardDeprecated();
    }

    // Destructor
    // Deletes all dependencies to this tensor
    // TODO
    ~Tensor() {
        // Go back towards computational graph
        // delete every Tensor and Op encountered in a DFS fashion
        delete backOp;
    }

};

#endif

