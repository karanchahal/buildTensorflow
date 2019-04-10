#include "buildTensorflow.h"

// Example of training a network on the buildTensorflow framework.
int main() {
    // Load Dataset
    Celsius2Faranheit<float,float> dataset;
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
            auto l = new Tensor<float>({-1}, {1,1});
            auto k = tensorOps::multiply(l,tar);
            auto loss = tensorOps::add(out,k); // error in loss
            auto finalLoss = tensorOps::power(loss,(float)2);

            // Compute backProp
            finalLoss->backward();
            // cout<<finalLoss->val<<endl;

            // Perform Gradient Descent
            sgd.minimise(finalLoss);
        
        }
    }

    cout<<"Training completed"<<endl;

    // Inference
    float cel = 4;
    auto test = new Tensor<float>({cel}, {1,1});
    auto out1 = fc1.forward(test);

    cout<<"The conversion of "<<cel<<" degrees celcius to faranheit is "<<out1->val<<endl; // For 4 Celcius: it's ~39.2

}

