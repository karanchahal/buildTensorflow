#include "types/tensor.h"

namespace utils {

    template< typename T>
    Tensor<T> initWeights(vector<int> shape) {
        int  p =1;
        for(auto i: shape) {
            p *= i;
        }
        vector<T> val(p,0);

        for(int i = 0;i< val.size();i++) {
            // TODO change init of weight to Glorot Initialisation
            // Make this extensible and open it up to the user for change
            int temp = 0;
            val[i] = temp;
        }
    }
}
