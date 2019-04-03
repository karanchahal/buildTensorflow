#include "types/tensor.h" 
#include "operations/operations_Impl.h"
#include "overloads/tensor.h"
#include <random>
#include <string>

#ifndef __DENSE_LAYER_INCLUDED__   
#define __DENSE_LAYER_INCLUDED__  

template<typename T>
class Dense{
    private:
    
    /*
        Glorot/ Xavier Initialisation. See paper:
        "Understanding the difficulty of training deep feedforward neural networks"
        by Bengio and Glorot
        Paper Link: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    
        TODO: Use a Uniform or a normal distribution ?
        Currently using a uniform distribution
    */
    vector<T> initWeights(int fan_in, int fan_out) {
        double variance = 2.0/(fan_in + fan_out);
        auto stddev = sqrt(variance);

        std::default_random_engine generator;
        std::uniform_real_distribution<T> distribution(0.0,stddev);

        vector<T> weights(fan_in*fan_out,0);
        for(int i = 0;i <fan_in*fan_out;i++) {
            T sample = distribution(generator);
            weights[i] = sample;
        }

        return weights;
    }

    public:
    Tensor<T>* weights; // size is input*output
    Tensor<T>* biases;
    int input_size;
    int output_size;
    string act;

    Dense(int input_size, int output_size, string activation) {
        this->input_size = input_size;
        this->output_size = output_size;

        auto weightVal = initWeights(input_size, output_size);
        this->weights = new Tensor<T>(weightVal,{input_size, output_size});
        this->biases = new Tensor<T>(vector<T>(output_size,0), {1, output_size});

        this->act = activation;
    }

    Tensor<T>* forward(Tensor<T> *x) {
        auto dot = tensorOps::dot(x,weights);
        Tensor<T>* logits;
        if(act == "sigmoid") {
            logits = tensorOps::sigmoid(dot);
        }
        auto z = tensorOps::add(logits, biases);
        return z;
    }

    void updateGradients() {
        // just update gradients of weights
    }
};

#endif
