/* 
    Utils for matrices. Functions like transpose and for future util matrix functions.
*/

#include "types/matrix.h"

#ifndef __UTILS_MATRIX_INCLUDED__   
#define __UTILS_MATRIX_INCLUDED__  

namespace utils {
    
    /* 
        Function to convert a Matrix to a transpose of it. 
        Just changes the shape vector of a matrix and return new matrix.
    */
    template< typename T>
    Matrix<T> transpose(const Matrix<T> &m, vector<int> transposeIndices) {
        assert(transposeIndices.size() == m.shape.size() && 
        "Transpose indices don't match shape for transpose operation");
        vector<int> shape(m.shape.size(),0);
        for(int i = 0;i<transposeIndices.size();i++) {
            shape[i] = m.shape[transposeIndices[i]];
        }

        return Matrix<T>(m.val,shape);
    }
}

#endif