#include "types/matrix.h"

#ifndef __TESTS_UTILS_INCLUDED__   
#define __TESTS_UTILS_INCLUDED__  

namespace testUtils{

    template<typename T>
    bool isMatrixEqual(Matrix<T> &lhs, Matrix<T> &rhs) {
        int n = lhs.shape.size();
        int m = rhs.shape.size();

        if(n != m) {
            return false;
        }

        for(int i = 0; i<n;i++) {
            if(lhs.shape[i] != rhs.shape[i]) {
                return false;
            }
        }

        n = lhs.val.size();

        for(int i = 0;i<n;i++) {
            if(lhs.val[i] != rhs.val[i]) {
                return false;
            }
        }
        
        return true;
    }
}

#endif
