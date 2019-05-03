/*
    This file defines the various mathematical overloads of the Matrix Class.
*/

#include <types/matrix.h>

#ifndef __MATRIX_OPS_INCLUDED__   
#define __MATRIX_OPS_INCLUDED__  


namespace matrixOps {

    // Sigmoid Operation
    template<typename T>
    Matrix<T> sigmoid(const Matrix<T> &a) {
        return (T)1/((T)1 + (((T)-1)*a).exp());
    }

    // Power Operation
    template<typename T>
    Matrix<T> power(Matrix<T> &a, T pow) {
        return a^pow;
    }

    template<typename T>
    Matrix<T> expandAlong(Matrix<T> &grad, int axis, int expansion) {
        vector<T> val;
        val.reserve(grad.val.size()*expansion);
        for(int i = 0;i < expansion;i++) {
            val.insert(val.end(), grad.val.begin(), grad.val.end());
        }
        auto s = grad.shape;
        s.insert(s.begin() + axis,expansion);
        return Matrix<T>(val, s);
    }

    template<typename T>
    Matrix<T> softmax(Matrix<T> &a, int axis) {
        auto e = a.exp();
        auto sum = e.addAxis(axis);
        auto ans = e/sum;
        return ans;
    }

    template<typename T>
    Matrix<T> gradSoftmax(Matrix<T> &a, int axis) {

        // auto one = a.exp();
        // auto sum = one.addAxis(1);

        // auto three =  ((float)-1) * (one - sum);
        // auto four = one * three;
        // auto five = four/sum;
        // auto six = five/sum;
        


        auto e = a.exp();
        auto sum = e.addAxis(axis);
        auto mult = e - sum;
        auto diff = ((T)-1) * mult;
        auto to_be_div = e * diff;
        auto one = to_be_div/sum;
        auto one_more = one/sum;

        auto t3 = softmax(a, axis);
        auto g = ((T)1 - t3) * t3; // (1 - F)*F

        // cout<<g<<endl;

        return one_more;
    }

    
};
// Overloaded function for printing matrix: cout<<matrix<<endl;
template<typename T>
ostream & operator << (ostream &out, Matrix<T> &m) {
    vector<int> stack;
    return m.print(out,stack,0);
}

// Divison with a scalar as divident
template<typename T>
Matrix<T> operator / (const T t, const Matrix<T> &rhs) {
    auto res =  t/rhs.val;
    auto resShape = rhs.shape;
    return Matrix<T>(res, resShape);
}

// Divison with a scalar as divident
template<typename T>
Matrix<T> operator / (const Matrix<T> &rhs, const T t) {
    auto res =  rhs.val/t;
    auto resShape = rhs.shape;
    return Matrix<T>(res, resShape);
}

// Multiplication with a scalar
template<typename T>
Matrix<T> operator * (const T t, const Matrix<T> &rhs) {
    auto res =  t*rhs.val;
    auto resShape = rhs.shape;
    return Matrix<T>(res, resShape);
}

// Addition with a scalar
template<typename T>
Matrix<T> operator + (const T t, const Matrix<T> &rhs) {
    auto res =  t+rhs.val;
    auto resShape = rhs.shape;
    return Matrix<T>(res, resShape);
}

// Subtraction with a scalar
template<typename T>
Matrix<T> operator - (const T t, const Matrix<T> &rhs) {
    auto res =  t-rhs.val;
    auto resShape = rhs.shape;
    return Matrix<T>(res, resShape);
}

#endif