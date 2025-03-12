#pragma once
#include <iostream>

#include <memory>
#include <algorithm>
#include <vector>

#include <fstream>
#include <utility>
#include <math.h>
#include <time.h>
#include <chrono>

#include <Eigen/Eigen>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "GpuArray.cuh"
#include "Timer.cuh"


using Funcptr = float (*) (const std::vector<float> &p1,const std::vector<float> &p2);
/*Base function*/
__host__ __device__ float F2_distance(const std::vector<float> &p1,const std::vector<float> &p2);
__host__ __device__ float F1_distance(const std::vector<float> &p1,const std::vector<float> &p2);
__host__ __device__ float Fn_distance(const std::vector<float> &p1,const std::vector<float> &p2);

enum class Type{
    FPS_NORMAL,
    FPS_PROCESSIVE
};


////////////////////////////////////////
//                                    //
//           Ifps Function            //
//                                    //
////////////////////////////////////////
class Ifps{
    
public:
    Ifps(){};

    /*
    * -fps : compute fps points from point clouds
    * -inverse fps: compute radius of fps points
    * -inverse shell: generate shells from fps points
    */

    // -fps
    std::vector<std::vector<float>> fps(std::vector<std::vector<float>> &dense_pts, const unsigned int m, const unsigned int normChoice);
    /* -Inverse Fps*/
    std::vector<float> inverse_fps(const std::vector<std::vector<float>> & ordered_pt,const unsigned int m,const unsigned int normChoice);
    /* -Inverse Shell*/
    std::vector<int> inverse_shell_cpu(const std::vector<std::vector<float>> &infer_pts, const std::vector<std::vector<float>> &ordered_pt,const std::vector<float> &ordered_radi,const unsigned int m,const unsigned int normChoice);
    std::vector<int> inverse_shell_gpu(const std::vector<Eigen::Vector3f> &infer_pts, const std::vector<Eigen::Vector3f> &ordered_pt,const std::vector<float> &ordered_radi, const unsigned int normChoice);
    torch::Tensor inverse_shell_gpu(const torch::Tensor &infer_pts,const torch::Tensor &ordered_pt,const torch::Tensor &ordered_radi,const unsigned int normChoice);

private:
    std::shared_ptr<Funcptr> funptr = std::make_shared<Funcptr>(&F2_distance);
};

