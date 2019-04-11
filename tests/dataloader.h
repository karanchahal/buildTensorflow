#include "data/celsius2fahrenheit.h"
#include <gtest/gtest.h>

/*
    This test trains a small neural network that learns how to convert celsius
    to fahrenheit. The test checks if the neural network trains successfully.

    We initiliase a neural network of 1 hidden neuron with no activation and
    train with mse loss.
*/
TEST(DATA_LOADER_TESTS, Celsius2FahrenheitDataLoaderTest) {
    // Load Dataset
    Celsius2Fahrenheit<float,float> dataset;
    dataset.create(5);

    for(auto example: dataset.data) {
        auto cel = example.first;
        auto tar = example.second;

        ASSERT_EQ(tar,9*cel/5 + 32);
    }


}