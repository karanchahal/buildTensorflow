/* 
    This file contains all the code to make a dense or a fully connected neural network layer. The API design is 
    meant to be pluggable meaning users can play around with different combinations of layers to construct their own
    network recipes.

    An exmaple usage of the dense layer is given as follows:

    Dense<float> fc1(2,5);
    Tensor<float>* x = new Tensor<float>({1,2},{1,2});
    auto m = fc1.forward(x);
    m->backward();
*/

#include "utils/weights.h"
#include "overloads/tensor.h"

#ifndef __SOFTMAX_LAYER_INCLUDED__   
#define __SOFTMAX_LAYER_INCLUDED__  

/*
    These specify enums for the vaious initialisation schemes and activation functions that can be used in 
    neural network layers. 

    TODO: Shift these to a common header file in the future
*/

enum activation{SIGMOID, RELU, NO_ACTIVATION}; 
enum initalisation{GLOROT};

template<typename T>
class Softmax{
    private:
 
    public:


    /*
        This constructor sets various variables and initialises the weights and biases matrices. It initilises the
        weights as per the initialisation scheme mentioned.

        By default activation used is Sigmoid and initialisation used is GLOROT
    */
    Softmax() {
       
    }

    /*
        Specifies the forward propogation of our dense layer. Basically generates the output for any given input for
        the given specification of the dense layer.

        For input x, it gets output y.
        Where, y = activation(weights@x + biases)

        Where @ represents dot product.
        other operations are all elementwise.
    */
    Tensor<T>* forward(Tensor<T> *x) {
        // get exponent 
        // add all exponent 
        // divide each by exponent 
        auto z = tensorOps::exp(x);
        auto l = tensorOps::add(z, axis=0); // add across an axis
        auto m = tensorOps::divide(z, l); // divide with a smaller tensor

        return m;

    }

    /*
        This function is temporary and is marked as TODO. When SGD or some gradient descent algorithm is called
        we might use this function to update the weights and biases of the dense layer according to a 
        certain learning rate.

        TODO
    */
    void updateGradients(float lr) {};
};

#endif
