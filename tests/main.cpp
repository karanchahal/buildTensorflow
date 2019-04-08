/*
    Driver file to run all tests in this folder
*/

// Add include statement of header files to run all tests below
#include <gtest/gtest.h>
#include "tests/matrix.h"
#include "tests/tensor.h"
#include "tests/dense.h"
#include "tests/sgd.h"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
