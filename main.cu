#include "buildTensorflow.h"
#include "mnist/include/mnist/mnist_reader_less.hpp"

template< typename T>
vector<T> getBatchedExamples(vector<vector<T>> &dataset, int index, int batch_size) {
    vector<T> ans;
    for(int i = index; i < index+batch_size;i++) {
        auto j = dataset[i];
        ans.insert(ans.end(), j.begin(), j.end());
    }
    return ans;
}

template< typename T>
vector<T> getBatchedTargets(vector<u_int8_t> &gt, int index, int batch_size, int num_classes) {
    vector<T> ans(batch_size*num_classes,0);
    for(int i = index; i < index+batch_size;i++) {
        auto j = gt[i];
        ans[(i-index)*num_classes + (int)j] = 1;
    }

    return ans;
}

void mnistTest() {
    
    // Load MNIST Dataset
    auto dataset = mnist::read_dataset<float, uint8_t>();
    auto train_images = dataset.training_images;
    auto train_labels = dataset.training_labels;
    auto test_images = dataset.test_images;
    auto test_labels = dataset.test_labels;


    // Create Model
    Dense<float> fc1(784, 200, SIGMOID);
    Dense<float> fc2(200, 200,SIGMOID);
    Dense<float> fc3(200, 10);

    // Initialise Optimiser
    SGD<float> sgd(0.001);

    int batch_size = 64;
    int num_classes = 10;
    int print_after = 100;

    // Train
    int num_examples = train_images.size();
    for(int j = 0;j<1;j++) {

        int ld = 0;
        float loss_till_now = 0;

        for(auto kl = 0; kl< num_examples; kl+=batch_size) {
            
            // Get Input
            auto i = getBatchedExamples(train_images, kl, batch_size);
            auto inp = new Tensor<float>(i, {batch_size,784});

            cout<<inp->val<<endl;
            cout<<tar->val<<endl;
            break;
            
            // Targets
            auto one_hot = getBatchedTargets<float>(train_labels, kl, batch_size, num_classes);
            auto tar = new Tensor<float>(one_hot, {batch_size,num_classes});
            
            // Forward Prop
            auto temp = fc1.forward(inp);
            auto temp2 = fc2.forward(temp);
            auto out = fc3.forward(temp2);
            
            // Get Loss
            auto finalLoss = losses::binary_cross_entropy(out, tar);
            
            // Compute backProp
            finalLoss->backward();
            cout<<finalLoss->val<<endl;
            // Perform Gradient Descent
            sgd.minimise(finalLoss);

            float h = 0;
            for(auto g: finalLoss->val.val) {
                h += g;
            }

            loss_till_now += h;

            if((kl+1)%(print_after*batch_size) == 0) {
                cout<<loss_till_now*batch_size/(kl+1)<<endl;
            }

            ld++;        
        }

        cout<<loss_till_now/num_examples<<endl;

    }

    // // // Inference
    // auto testVal = train_images[0];
    // auto test = new Tensor<float>({testVal}, {1,784});
    // auto temp = fc1.forward(test);
    // auto temp2 = fc2.forward(temp);
    // auto ans = tensorOps::softmax(fc3.forward(temp2), 1);


    // cout<<ans->val<<endl;

    // // cout<<ans->val<<endl;
    // cout<<(float)train_labels[0]<<endl;

    // delete ans;
}


// Example of training a network on the buildTensorflow framework.
int main() {

   mnistTest();
    // Matrix<float> a({1,2,3,4,5,6}, {2,3});
    // Matrix<float> b({1,2,3}, {3});

    // auto c = a+b;
    // cout<<c<<endl;

}
