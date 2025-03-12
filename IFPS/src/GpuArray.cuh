#pragma once 
#include <memory>
#include <vector>
#include <iostream>

#define CUDA_CHECK_THROW(call){                                         \
     cudaError_t cudaStatus = call;                                     \
     if(cudaStatus != cudaSuccess){                                     \
        std::cerr<< " CUDA error: " << cudaGetErrorString(cudaStatus);  \
        std::exit(EXIT_FAILURE);                                        \
     }                                                                  \
}

template <typename T>
class GpuArray{
private:
    T*  _data = nullptr;
    size_t _size;

public:
    GpuArray(){};
    GpuArray(size_t size):_size(size){
        // Allocate memory on GPU
        CUDA_CHECK_THROW(cudaMalloc(&_data,_size * sizeof(T)));
    }
    GpuArray(const T& hostData,cudaStream_t stream = nullptr){
        // Allocate memory on GPU
        CUDA_CHECK_THROW(cudaMalloc(&_data,  sizeof(T)));
        // Copy data from host to device(CPU to GPU)
        CUDA_CHECK_THROW(cudaMemcpyAsync(_data,hostData.data(), sizeof(T),cudaMemcpyHostToDevice,stream));
    }

    GpuArray(const std::vector<T> &hostData,size_t size,cudaStream_t stream = nullptr):_size(size){
        // Allocate memory on GPU
        CUDA_CHECK_THROW(cudaMalloc(&_data,size * sizeof(T)));
        // Copy data from host to device(CPU to GPU)
        CUDA_CHECK_THROW(cudaMemcpyAsync(_data,hostData.data(),size * sizeof(T),cudaMemcpyHostToDevice,stream));
    }

    ~GpuArray(){
        CUDA_CHECK_THROW(cudaFree(_data));
    }


    T *ptr(){
        return _data;
    }



    void copyToCPU(T *hostData,size_t size){
        CUDA_CHECK_THROW(cudaMemcpyAsync(hostData,_data,size * sizeof(T),cudaMemcpyDeviceToHost));
    }

    
};
