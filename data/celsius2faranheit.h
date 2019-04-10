/*
    This file defines the Celsius To Faranheit DataLoader. It's input is variables containing the
    celsius numbers and the targets are the corresponding faranheit numbers.

    The way to use this dataset is as follows:

    Celsius2Faranheit<float,float> dataloader;
    dataloader.create(10); // Creates 10 training examples

    for(auto i: dataloader.data) {
        auto inp = j.first;
        auto tar = j.second;

        // And then use this above data in your model for training or inference
    }

    Note that the data won't be outputted in tensors. It will simply be of the data type the user 
    signifies in the dataloader defination. In the above case the input and targets are both floats.

*/

#include "data/dataloader.h"
#include <stdlib.h>

#ifndef __C2F_DATASET_INCLUDED__   
#define __C2F_DATASET_INCLUDED__ 

template<typename I, typename T>
class Celsius2Faranheit: public DataLoader<I,T> {
    
    private:
    int MAX_CELSIUS = 10;

    // Helper function to convert celsius to faranheit
    T toFaranheit(I input) {
        return (9*input)/5 + 32;
    }

    public: 

    // Adds a training example into the dataset
    void add(I input, T target) {
        this->data.push_back(make_pair(input,target));
    }

    // Populates the dataset with the number of examples specified by the user.
    void create(int num_examples) {
        for(int i=0; i< num_examples;i++) {
            I input = rand() % MAX_CELSIUS + 1; // random int value between 1 and MAX_CELSIUS
            T target = toFaranheit(input);
            add(input,target);
        }
    }
};

#endif
