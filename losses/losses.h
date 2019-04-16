#include "types/tensor.h"
#include "utils/matrix.h"


#ifndef __LOSSES_INCLUDED__   
#define __LOSSES_INCLUDED__  

namespace losses {
    
    /*
        Mean Squared Error Loss. The inputs and targets have to be of the same shape. The loss computes the mean
        square error between the output of the network and ground truth.

        The formula is as follows:
        loss = (y - ground_truth)^2.

        This loss can be used for regression problems.
 
    */
    template<typename T>
    Tensor<T>* mse(Tensor<T>* y, Tensor<T>* ground_truth) {
        assert(y->val.shape == ground_truth->val.shape && "Shapes of ground truth and output don't match for computing loss!");
        
        auto v = ((T)-1)*utils::onesLike(ground_truth->val);
        auto l = new Tensor<T>(v);  // tensor of -1s

        // TODO: Replace with subtraction operation
        auto k = tensorOps::multiply(l,ground_truth);

        auto loss = tensorOps::add(y,k); // error in loss
        auto finalLoss = tensorOps::power(loss,(T)2); // mean squared error

        return finalLoss;
    }
    
    /*
        This loss is given by (1-gt)*log(1-y) + gt*log(y).

        Here we assume that gt is a one hot vector signifying 0 for negative class and 1 for positive class.
        Also, the value of y will be between 1 and zero. Most liklely it comes out of a sigmoid or a softmax distribution.
        loss[i] = -log(softmax(x[i])) = -x[i] + log(sum(x)) 
    */
    template<typename T>
    Tensor<T>* binary_cross_entropy(Tensor<T>* y, Tensor<T>* ground_truth) {

        auto probs = tensorOps::softmax(y,1);
        auto z = tensorOps::multiply(probs, ground_truth);
        auto z2 = tensorOps::add(z,1);
        auto z3 = tensorOps::log(z2);
        auto z4 = tensorOps::multiply((T)-1, z3);
        auto z5 = tensorOps::average(z4,0);

        return z5;
    }
}

#endif
