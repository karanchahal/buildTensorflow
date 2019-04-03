#include "types/tensor.h" 
#include "operations/operations_Impl.h"
#include "overloads/tensor.h"
#include <random>
#include <string>

#ifndef __BATCHNORM_LAYER_INCLUDED__   
#define __BATCHNORM_LAYER_INCLUDED__  

/*
Batch Normalization: Accelerating Deep Network Training by Reducing 
Internal Covariate Shift

Paper Link: https://arxiv.org/pdf/1502.03167.pdf
*/
template<typename T>
class BatchNorm{
    private:

    std::tuple<T, T> calcMeanVariance(Tensor<T>* x) {
        // get scalar value of mean
        T mean = 0
        int n = x->val.size();
        for(int i = 0;i<n;i++) {
            mean += x->val[i];
        }
        mean = mean/n;
        T variance = 0
        for(int i = 0;i<n;i++) {
            variance += sqrt(x->val[i] - mean);
        }

        variance = variance/n;
        // get scalar value of variance
        return {mean,variance};
    }
    
    Tensor<T> gamma, beta;
    public:

    BatchNorm() {
        // init gamma and beta
    }

    Tensor<T>* forward(Tensor<T>* x) {
        auto [variance, mean] = calcMeanVariance(x);

        auto y = (x-mean)/(variance);
        auto z = gamma*y + beta;

        return z;
    }

    void updateGradients() {
        // only update gamma and beta gradients
        // in this layer

    }
};

#endif
