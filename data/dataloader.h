#include<iostream>
#include<vector>

#ifndef __DATALOADER_INCLUDED__   
#define __DATALOADER_INCLUDED__ 

template<typename I, typename T>
class DataLoader {

    public: 
    vector<pair<I,T>> dataset;

    virtual void add(I input, T target) = 0;
};

#endif
