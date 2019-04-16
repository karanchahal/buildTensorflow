#include "buildTensorflow.h"
#include "mnist/include/mnist/mnist_reader_less.hpp"

void mnistTest() {
     // Load MNIST Dataset
    auto dataset = mnist::read_dataset<float, uint8_t>();
    auto train_images = dataset.training_images;
    auto train_labels = dataset.training_labels;
    auto test_images = dataset.test_images;
    auto test_labels = dataset.test_labels;


    // Create Model
    Dense<float> fc1(784, 100);
    Dense<float> fc2(100, 100,SIGMOID);
    Dense<float> fc3(100, 20,SIGMOID);
    Dense<float> fc4(20, 10, SIGMOID);

    // Initialise Optimiser
    SGD<float> sgd(0.001);
    
    // Train
    int num_examples = 2;
    for(int j = 0;j<4000;j++) {
        int ld = 0;
        float loss_till_now = 0;
        for(auto kl = 0; kl< num_examples;kl++) {
            auto i = train_images[kl];
            // Get data
            auto inp = new Tensor<float>({i}, {1,784});
            vector<float> one_hot(10,0);
            one_hot[(int)train_labels[ld]] = 1;
            auto tar = new Tensor<float>(one_hot, {1,10});

            // Forward Prop
            auto temp = fc1.forward(inp);
            auto temp2 = fc2.forward(temp);
            auto temp3 = fc3.forward(temp2);
            auto out = fc4.forward(temp3);
            
            // Get Loss
            auto finalLoss = losses::mse(out, tar);

            // Compute backProp
            finalLoss->backward();

            // Perform Gradient Descent
            sgd.minimise(finalLoss);
            float h = 0;
            for(auto g: finalLoss->val.val) {
                h += g;
            }
            loss_till_now += h;

            ld++;        
        }

        cout<<loss_till_now/num_examples<<endl;

    }

    // // Inference
    auto testVal = train_images[0];
    auto test = new Tensor<float>({testVal}, {1,784});
    auto temp = fc1.forward(test);
    auto temp2 = fc2.forward(temp);
    auto temp3 = fc3.forward(temp2);
    auto ans = fc4.forward(temp3);

    cout<<ans->val<<endl;

    // cout<<ans->val<<endl;
    cout<<(float)train_labels[0]<<endl;

}


// Example of training a network on the buildTensorflow framework.
int main() {

    auto a  = new Tensor<float>({1, 2, 3, 4}, {2,2});
    auto gt = new Tensor<float>({1, 0, 0, 1 }, {2,2});

    auto ans  = losses::binary_cross_entropy(a,gt);
    cout<<ans->val<<endl;
    // // so what we need to do is broadcasting support
    // b should become {3,3,2,2}, {2,2}

    // so remove any 1's from second matrix shape
    // check if shape is compatible that means, 
    // run it through front to back, it is fine if there is no shape mismatch
    // either shape has ended

    // once we have done this

    // then "copy" elements over to new matrix that is of same size as lhs

    // start from 
}
