#include "buildTensorflow.h"
#include <gtest/gtest.h>
#include "tests/utils.h"

/*
    This test trains a small neural network that learns how to convert celsius
    to fahrenheit. The test checks if the neural network trains successfully.

    We initiliase a neural network of 1 hidden neuron with no activation and
    train with mse loss.
*/
TEST(TRAINING_TEST, Celsius2FahrenheitTest) {
    
    // Load Dataset
    Celsius2Fahrenheit<float,float> dataset;
    dataset.create(5);

    // Create Model
    Dense<float> fc1(1,1,NO_ACTIVATION);

    // Initialise Optimiser
    SGD<float> sgd(0.01);
    
    // Train
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

            // Perform Gradient Descent
            sgd.minimise(finalLoss);
        
        }
    }

    // Inference
    float cel = 4;
    auto test = new Tensor<float>({cel}, {1,1});
    auto ans = fc1.forward(test);

    auto res = Matrix<float>({39.1999588},{1,1});
    cout<<res<<endl;
    cout<<ans->val<<endl;
    ASSERT_TRUE(testUtils::isMatrixEqual(ans->val,res));

    // Clean up
    delete ans;

}
