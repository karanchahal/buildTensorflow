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

// #include "types/tensor.h" 
// #include "operations/operations_Impl.h"
// #include "overloads/tensor.h"
#include "utils/weights.h"

#ifndef __DENSE_LAYER_INCLUDED__   
#define __DENSE_LAYER_INCLUDED__  

/*
    These specify enums for the vaious initialisation schemes and activation functions that can be used in 
    neural network layers. 

    TODO: Shift these to a common header file in the future
*/

enum activation{SIGMOID, RELU, NO_ACTIVATION}; 
enum initalisation{GLOROT};

template<typename T>
class Dense{
    private:
    
    vector<T> initWeights(int fan_in, int fan_out, initalisation init) {
        if(init == GLOROT) {
            return utils::glorotInit<T>(fan_in, fan_out);
        }

        // Default return zero vector
        return vector<T>(fan_in*fan_out,0);
    }

    public:

    // Weights are of shape input_size by output_size, hence they are a 2D matrix
    Tensor<T>* weights;
    
    // Biases are a 1D matrix of size output_size
    Tensor<T>* biases;

    // Specifies input and output size of our dense layer. 
    int input_size, output_size;

    // Specifies type of activation to be used in the dense layer.
    activation act;

    /*
        This constructor sets various variables and initialises the weights and biases matrices. It initilises the
        weights as per the initialisation scheme mentioned.

        By default activation used is Sigmoid and initialisation used is GLOROT
    */
    Dense(int input_size, int output_size, activation act=SIGMOID, initalisation init=GLOROT) {
        this->input_size = input_size;
        this->output_size = output_size;

        auto weightVal = initWeights(input_size, output_size, init);
        this->weights = new Tensor<T>(weightVal,{input_size, output_size});
        this->biases = new Tensor<T>(vector<T>(output_size,0), {1, output_size});

        this->act = act;
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

        auto dot = tensorOps::dot(x,weights);
        auto z = tensorOps::add(dot, biases);
        Tensor<T>* logits;
        
        if(act == SIGMOID) {
            logits = tensorOps::sigmoid(z);
        } else if(act == NO_ACTIVATION) {
            logits = z;
        }
        
        return logits;
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
