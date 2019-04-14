#ifndef __PROJECT_INCLUDED__   
#define __PROJECT_INCLUDED__ 

// Check whether GPU is accessible or not
#ifndef __GPU_INCLUDED__   
#define __GPU_INCLUDED__ 
bool gpu = true;
#endif

#include "gpu/defn.h" // Includes GPU Kernel Code Defination for Forward pass
#include "types/tensor.h"
#include "gpu/impl.h" // Includes GPU Kernel Code Implementation
#include "overloads/tensor.h"
#include "operations/operations_Impl.h"
#include "layers/dense.h"
#include "optims/sgd.h"

#endif
