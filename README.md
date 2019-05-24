# BuildTensorflow

A lightweight deep learning framework made with ❤️

## Introduction

Ever interacted with code that looks like this ?

```c++
int main() {
    // Load Dataset
    Celsius2Fahrenheit<float,float> dataset;
    dataset.create(5);

    // Create Model
    Dense<float> fc1(1,1,NO_ACTIVATION);

    // Initialise Optimiser
    SGD<float> sgd(0.01);
    
    // Train
    cout<<"Training started"<<endl;
    for(int j = 0;j<2000;j++) {
        for(auto i: dataset.data) {
            // Get data
            auto inp = new Tensor<float>({i.first}, {1,1});
            auto tar = new Tensor<float>({i.second}, {1,1});

            // Forward Prop
            auto out = fc1.forward(inp);

            // Get Loss
            auto finalLoss = tensorOps::mse(tar,out);

            // Compute backProp
            finalLoss->backward();

            // Perform Gradient Descent
            sgd.minimise(finalLoss);
        
        }
    }

    cout<<"Training completed"<<endl;

    // Inference
    float cel = 4;
    auto test = new Tensor<float>({cel}, {1,1});
    auto out1 = fc1.forward(test);

    cout<<"The conversion of "<<cel<<" degrees celsius to fahrenheit is "<<out1->val<<endl; // For 4 Celcius: it's ~39.2
}
```


No, this isn't some C++ API from Pytorch, this is our very own lightweight deep learning framework learning how to become a celsius to fahrenheit convertor ! The whole codebase is less than a 1000 lines of code and has no external dependencies.

Also, did we mention that our neural network can also be run on the GPU ? (you'll need CUDA support for this though) 

## Ugh, why do we need another Deep Learning framework ?

Have you every wanted to know how ```loss.backward()``` works in your Pytorch code? Or what does sgd.minimise(loss) even do ?

Sure you've read the theory, you know _how_ it works but you don't really know __how__ it works.

Have you ever wanted to go looking into the Pytorch codebase trying to find how automatic differentiation works ? I've tried to and have found the experience really stressful. Diving into those many lines of code is mentally draining and leaves you with more questions that answers. 

And, that is why this project exists. 

We want to give readers interested in deep learning frameworks an idea of how everything works under the hood by providing a clear and concise codebase that is less than 1000 lines of code, expertly documented and tested rigorously.

## Goals

In this learning odyssey we hope to give readers the knowledge of the following:

1. How production code is written in C++ along with how it is structured. We have written unit tests for each feature and have commented each part of the codebase so that the reader is able to understand the code with minimal fuss.

2. Lightweight implementations of popular concepts such as Stochastic Gradient Descent, various Loss functions and how they interact with automatic differentiation.

3. How to speed up your neural networks by running matrix multiplications on the GPU. This framework is both CPU and GPU compatible and gives users some insight into how exactly code is parallelised on the GPU.
