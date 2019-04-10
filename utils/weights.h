/* 
    Utils for weights. Different weight initialisation schemes and for debugging weights in the future
*/


#include "types/tensor.h"
#include <random>

#ifndef __UTILS_WEIGHTS_INCLUDED__   
#define __UTILS_WEIGHTS_INCLUDED__  

namespace utils {

    /*
        Glorot/ Xavier Initialisation. See paper:
        "Understanding the difficulty of training deep feedforward neural networks"
        by Bengio and Glorot
        Paper Link: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    
        TODO: Use a Uniform or a normal distribution ?
        Currently using a uniform distribution
    */
    template< typename T>
    vector<T> glorotInit(int fan_in, int fan_out) {
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
}

#endif