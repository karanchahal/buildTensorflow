/*
    This file defines the base class of each Dataset in the project. The data is stored in a simple
    vector and each object in the vector is a pari signifying input and target (ground truth).
*/

#include<iostream>
#include<vector>

#ifndef __DATALOADER_INCLUDED__   
#define __DATALOADER_INCLUDED__ 

template<typename I, typename T>
class DataLoader {

    public: 
    // This variable contains all the data of the dataset
    vector<pair<I,T>> data;

    // This function perfroms the operation that populates the "data" variable.
    virtual void add(I input, T target) = 0;
}; 

#endif
