/* This file defines the Stochastic Gradient Descent Optimiser */

#include "types/tensor.h" 
#include "operations/operations_Impl.h"
#include "overloads/tensor.h"
#include <random>
#include <string>


/* 
    All formulas are referred from Sebastian Ruder's write up on optimisation
    algorithms

    Link: http://ruder.io/optimizing-gradient-descent/
*/
template<typename T>
class SGD {

    public:

    float lr;
    SGD(float lr) {
        this->lr = lr;
    }

    void minimize(Tensor<T>* loss) {

        queue<Tensor<T>*> q;
        q.push(loss);
        set<Tensor<T>*> cache;
        while(!q.empty()) {
            auto v = q.pop();
            if (!cache[v]) {

                cache.push(v);

                v = v - lr*v->grad;
                auto t1 = v->backOp->t1;
                auto t2 = v->backOp->t2;
                if(t1 != NULL) {
                    q.push(t1);
                }

                if(t2 != NULL) {
                    q.push(t2);
                }
            }
        }

    }
};