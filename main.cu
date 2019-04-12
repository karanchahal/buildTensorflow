#include "buildTensorflowGpu.h"
#include "mnist/include/mnist/mnist_reader_less.hpp"
// Example of training a network on the buildTensorflow framework.
int main() {

    // // Load MNIST Dataset
    // auto dataset = mnist::read_dataset<float, uint8_t>();
    // auto train_images = dataset.training_images;
    // auto train_labels = dataset.training_labels;
    // auto test_images = dataset.test_images;
    // auto test_labels = dataset.test_labels;


    // // Create Model
    // Dense<float> fc1(784,100);
    // Dense<float> fc2(100,20);
    // Dense<float> fc3(20,1, NO_ACTIVATION);

    // // Initialise Optimiser
    // SGD<float> sgd(0.0001);
    
    // // Train

    // float loss_till_now = 0;
    // for(int j = 0;j<1;j++) {
    //     int ld = 0;
    //     for(auto i: train_images) {
    //         // Get data
    //         auto inp = new Tensor<float>({i}, {1,784});
    //         auto tar = new Tensor<float>({(float)train_labels[ld]}, {1,1});

    //         // Forward Prop
    //         auto temp = fc1.forward(inp);
    //         auto temp2 = fc2.forward(temp);
    //         auto out = fc3.forward(temp2);

    //         // Get Loss
    //         auto l = new Tensor<float>({-1}, {1,1});
    //         auto k = tensorOps::multiply(l,tar);
    //         auto loss = tensorOps::add(out,k); // error in loss
    //         auto finalLoss = tensorOps::power(loss,(float)2);

    //         // Compute backProp
    //         finalLoss->backward();

    //         // Perform Gradient Descent
    //         sgd.minimise(finalLoss);
    //         loss_till_now += finalLoss->val.val[0];

    //         if(ld%2000 == 0) {
    //             cout<<loss_till_now/ld<<endl;
    //         }

    //         ld++;
        
    //     }
    // }

    // // Inference
    // auto testVal = test_images[0];
    // auto test = new Tensor<float>({testVal}, {1,784});
    // auto temp = fc1.forward(test);
    // auto temp2 = fc2.forward(temp);
    // auto ans = fc3.forward(temp2);

    // cout<<ans->val<<endl;
    // cout<<(float)test_labels[0]<<endl;

    // // ASSERT_TRUE(testUtils::isMatrixEqual(ans->val,res));

    // // // Clean up
    // // delete ans;
    
}

