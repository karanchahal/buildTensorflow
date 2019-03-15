#include "utils/common.h"
#include <cmath>

#ifndef __VEC_OVERLOADS_FLOAT_INCLUDED__   
#define __VEC_OVERLOADS_FLOAT_INCLUDED__  

// Multiplication
template<typename T>
vector<T> operator * (vector<T> &a, const vector<T> &b) {
    assert("Tensors are not of the same size !" && a.size() == b.size());
    vector<T> arr;
    for(int i = 0;i<a.size();i++) {
        T prod = a[i]*b[i];
        arr.push_back(prod);
    }
    return arr;
}

// Addition
template<typename T>
vector<T> operator + (vector<T> &a, const vector<T> &b) {
    assert("Tensors are not of the same size !" && a.size() == b.size());
    vector<T> arr;
    for(int i = 0;i<a.size();i++) {
        T prod = a[i]+b[i];
        arr.push_back(prod);
    }
    return arr;
}

// Vector Divide
template<typename T>
vector<T> operator / (vector<T> &a, const vector<T> &b) {
    assert("Tensors are not of the same size !" && a.size() == b.size());
    vector<T> arr;
    for(int i = 0;i<a.size();i++) {
        T prod = a[i]/b[i];
        arr.push_back(prod);
    }
    return arr;
}

// Scalar divide
template<typename T>
vector<T> operator / (T a, const vector<T> &b) {
    vector<T> arr;
    for(int i = 0;i<b.size();i++) {
        T prod = a/b[i];
        arr.push_back(prod);
    }
    return arr;
}

// Power Operation
template<typename T>
vector<T> operator ^ (vector<T> &a, const T b) {
    vector<T> arr;
    for(int i = 0;i<a.size();i++) {
        T prod = pow(a[i], b);
        arr.push_back(prod);
    }
    return arr;
}

// Expoenent Operation
template<typename T>
vector<T> exponent(const vector<T> &a) {

    vector<T> arr;
    for(int i = 0;i<a.size();i++) {
        T prod = exp(a[i]);
        arr.push_back(prod);
    }
    return arr;
}

// isEquals operator
template<typename T>
bool operator == (const vector<T> &a, const vector<T> &b) {
    int n = a.size();
    int m = b.size();

    if(n != m) {
        return false;
    }

    for(int i=0;i<n;i++) {
        if(a[i] != b[i]) {
            return false;
        }
    }

    return true;
}

#endif

