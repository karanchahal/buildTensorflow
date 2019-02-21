#include<vector>
#include<iostream>
#include <chrono> // for measuring performance
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

auto cpuVectorAddition(std::vector<int> &A, std::vector<int> &B) {
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0;i< A.size();i++) {
        A[i] += B[i];
    }
    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
    std::cout <<"Speed of CPU vector Addition: " << duration.count() <<" micro seconds"<<std::endl; 
    return duration.count();
}

__global__ void add(int *a, int *b, int*c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

// GPU vector Addition using Pointers
auto gpuVectorAddition(std::vector<int> &A, std::vector<int> &B) {
    size_t n= A.size();

    int* h_A = A.data();
    int* h_B = B.data();

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(int)*n);
    cudaMalloc((void**)&d_b, sizeof(int)*n);
    cudaMalloc((void**)&d_c, sizeof(int)*n);

    cudaMemcpy((void *)d_a, h_A, sizeof(int)*n, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_b, h_B, sizeof(int)*n, cudaMemcpyHostToDevice);

    // Timing stuff, record how many seconds it takes for this operation
    cudaEvent_t launch_begin, launch_end;
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);

    // Warmup
    add<<<n,1>>>(d_a, d_b, d_c);// num blocks, num_threads
    float total_time = 0;
    // Get average of 100 runs
    for(int i = 0;i<100;i++) {
        cudaEventRecord(launch_begin,0);
        add<<<n,1>>>(d_a, d_b, d_c);
        cudaEventRecord(launch_end,0);
        cudaEventSynchronize(launch_end);

        float time = 0;
        cudaEventElapsedTime(&time, launch_begin, launch_end);
        total_time += time;
    }

    total_time /= 100;
    std::cout <<"Speed of GPU vector Addition: " << total_time <<" micro seconds"<<std::endl; 

    // Copy memory back and free stuff
    cudaMemcpy((void *)h_A, d_c, sizeof(int)*n, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return total_time;
}

void vectorAdditionTest() {
    size_t n = 10000000;
    // std::cout<<n<<std::endl;

    std::vector<int> A(n,1);
    std::vector<int> B(n,-1);
    auto timeCpu = cpuVectorAddition(A,B);
    auto timeGpu = gpuVectorAddition(A,B);

    std::cout<<"Speedup over CPU is: "<< (float)timeCpu/timeGpu <<std::endl;
}

// Observation for CPU vs GPU compute in vector addition, the answer why is as follows:

// 1. CUDA has a start-up overhead. For "small" problems like this one, the startup overhead will outweigh any gains from using the GPU. 

int main() {

    // std::vector<int> A = {1,2,3};
    // std::vector<int> B = {3,4,5};

    // vectorAddition(A,B); // puts result in A
    // for(auto i: A) {
    //     std::cout<<i<<std::endl;
    // }

    vectorAdditionTest();

    return 1;
    
}