#ifndef __GPU_DOT_INCLUDED__   
#define __GPU_DOT_INCLUDED__  

template<typename T>
struct Matrix;

template<typename T>
void dotGPU(vector<T> &res, const Matrix<T>* lhs, const Matrix<T> &rhs, int start, int startRes);

#endif

