#include<iostream>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include<vector>

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

    A.assign(h_A, h_A + n);
    for(auto i: A) {
        if(i != 0) {
            std::cout<<i<< " False"<<std::endl;
            break;
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return total_time;
}

void vectorAdditionTest() {
    size_t n = 100000;
    // std::cout<<n<<std::endl;

    std::vector<int> A(n,1);
    std::vector<int> B(n,-1);
    auto timeCpu = cpuVectorAddition(A,B);
    auto timeGpu = gpuVectorAddition(A,B);

    std::cout<<"Speedup over CPU is: "<< (float)timeCpu/timeGpu <<std::endl;
}

// Observation for CPU vs GPU compute in vector addition, the answer why is as follows:

// 1. CUDA has a start-up overhead. For "small" problems like this one, the startup overhead will outweigh any gains from using the GPU. 


__global__ void mm(int* a, int* b, int* c, int width) {

    int x = blockIdx.x; // block id
    int y = threadIdx.x; // thread id
    int temp = 0;
    for(int i = 0;i< width;i++) {
        temp += a[x*width + i]*b[i*width+ y];
    }

    c[x*width + y] = temp;
}



// GPU vector Addition using Pointers
auto gpuMatrixMultiplication(std::vector<int> &A, std::vector<int> &B, int size, bool print) {
    
    size_t n= A.size();

    int* h_A = A.data();
    int* h_B = B.data();

    int *d_a, *d_b, *d_c;
    int* h_C = (int *)malloc(sizeof(int)*n);
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
    
    mm<<<size,size>>>(d_a, d_b, d_c,size);// num blocks, num_threads
    float total_time = 0;
    int num_times = 10;
    if(!print){
        // Get average of 100 runs
        for(int i = 0;i<num_times;i++) {
            cudaEventRecord(launch_begin,0);
            mm<<<size,size>>>(d_a, d_b, d_c, size);
            cudaEventRecord(launch_end,0);
            cudaEventSynchronize(launch_end);
            float time = 0;
            cudaEventElapsedTime(&time, launch_begin, launch_end);
            total_time += time;
        }
        
    }

    total_time /= num_times;

    // Copy memory back and free stuff
    cudaMemcpy(h_C, (void **)d_c, sizeof(int)*n, cudaMemcpyDeviceToHost);

    if(print) {
        for(int i = 0;i < n;i++) {
            std::cout<<h_C[i]<<" ";
        }
        std::cout<<std::endl;
    } else {
        std::cout <<"Speed of GPU vector Multiplication: " << total_time <<" micro seconds"<<std::endl; 
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return total_time;
}

float mmCpu(std::vector<int> &a, std::vector<int> &b, int n,bool print) {

    std::vector<int> c(n*n,0);
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0;i<n;i++) {
        for(int j = 0;j< n;j++) {
            for(int k = 0;k < n;k++) {
                c[i*n + k] += a[i*n + j] * b[j*n + k];
            }
        }
    }

    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 

    if(print) {
        for(auto i: c) {
            std::cout<<i<<" ";
        }
        std::cout<<std::endl;
    } else {
        std::cout <<"Speed of CPU vector Multiplication: " << duration.count() <<" micro seconds"<<std::endl; 
    }

    return duration.count();
}

void matrixMultiplySpeedTest() {
    int size = 1024;
    std::vector<int> A(size*size);
    std::vector<int> B(size*size);
    auto gpuSpeed = gpuMatrixMultiplication(A,B,size,false);
    auto cpuSpeed = mmCpu(A,B,size,false);

    std::cout<<"Speed of GPu over CPU is: "<< cpuSpeed/gpuSpeed<<std::endl;
}


void matrixMultiplyCorrectness() {
    int size = 4;
    std::vector<int> A = {3,1,2,4,3,1,2,4,3,1,2,4,3,1,2,4};
    std::vector<int> B = {3,1,2,4,3,1,2,4,3,1,2,4,3,1,2,4};
    gpuMatrixMultiplication(A,B,size,true);
    mmCpu(A,B,size,true);

}
int main() {

    matrixMultiplySpeedTest();
    matrixMultiplyCorrectness();
    return 1; 
}
