#include "types/matrix.h"

namespace utils {

    template< typename T>
    Matrix<T> transpose(Matrix<T> m, vector<int> transposeIndices) {
        assert(transposeIndices.size() == m.shape.size() && 
        "Transpose indices don't match shape for transpose operation");
        vector<int> shape(m.shape.size(),0);
        for(int i = 0;i<transposeIndices.size();i++) {
            shape[i] = m.shape[transposeIndices[i]];
        }

        return Matrix<T>(m.val,shape);
    } 
}
