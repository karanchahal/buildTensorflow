// Check whether GPU is accessible or not
bool gpu = true;

// #include "gpu/defn.h" // Includes GPU Kernel Code Defination for Forward pass
#include "types/tensor.h"
// #include "gpu/impl.h" // Includes GPU Kernel Code Implementation
#include "overloads/tensor.h"
#include "operations/operations_Impl.h"
#include "layers/dense.h"
#include "optims/sgd.h"