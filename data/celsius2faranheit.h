#include "data/dataloader.h"
#include <stdlib.h>

#ifndef __C2F_DATASET_INCLUDED__   
#define __C2F_DATASET_INCLUDED__ 

template<typename I, typename T>
class Celsius2Faranheit: public DataLoader<I,T> {
    
    private:
    int MAX_CELSIUS = 10;

    T toFaranheit(I input) {
        return (9*input)/5 + 32;
    }

    public: 
    void add(I input, T target) {
        this->dataset.push_back(make_pair(input,target));
    }

    void create(int num_examples) {
        for(int i=0; i< num_examples;i++) {
            I input = rand() % MAX_CELSIUS + 1; // random int value between 1 and MAX_CELSIUS
            T target = toFaranheit(input);
            add(input,target);
        }
    }
};

#endif
