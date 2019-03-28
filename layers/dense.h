// TODO: 
// 1. Init dense layer
// Basically a weight matrix,
// Feed in input, get output
// Write back 
#include "buildTensorflow.h"

template<typename T>
class Dense{
    private:
    
    /*
        Glorot/ Xavier Initialisation. See paper:
        "Understanding the difficulty of training deep feedforward neural networks"
        by Bengio and Glorot
        Paper Link: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    
        TODO: Use a Uniform or a normal distribution ?
    */
    vector<T> initWeights(int fan_in, int fan_out) {
        double variance = 2.0/(fan_in + fan_out);
        auto stddev = sqrt(variance);

        default_random_engine generator;
        normal_distribution<T> distribution(0.0,stddev);

        vector<T> weights(0,fan_in*fan_out);
        for(int i = 0;i <fan_in*fan_out;i++) {
            weights[i] = distribution(generator);
        }

        return weights;
    }

    public:
    Tensor<T>* weights; // size is input*output
    Tensor<T>* biases;
    int input_size;
    int output_size;

    Dense(int input_size, int output_size) {
        this->input_size = input_size;
        this->output_size = output_size;

        auto weightVal = initWeights(input_size, output_size);
        this->weights = new Tensor<T>(weightVal,{input_size, output_size});
        this->biases = new Tensor<T>(vector<T>(output_size,0), {output_size});
    }

    Tensor<T>* forward(Tensor<T> *x) {
        auto dot = tensorOps::dot(x,weights);
        auto logits = tensorOps::sigmoid(dot);
        auto z = tensorOps::add(logits, biases);
        return z;
    }
}