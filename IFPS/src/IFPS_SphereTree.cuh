#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <Eigen/Eigen>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "GpuArray.cuh"
#include "Timer.cuh"


__host__ __device__ float dist(const Eigen::Vector3f a,const Eigen::Vector3f b)
{
	return ((b[0] - a[0]) * (b[0] - a[0])) 
    + ((b[1] - a[1]) * (b[1] - a[1])) 
    + ((b[2] - a[2]) * (b[2] - a[2]));
}

__host__ __device__ float dist(const float * a,const float * b)
{
	return ((b[0] - a[0]) * (b[0] - a[0])) 
    + ((b[1] - a[1]) * (b[1] - a[1])) 
    + ((b[2] - a[2]) * (b[2] - a[2]));
}

////////////////////////////////////////
//                                    //
//     Ifps SphereTree Function       //
//                                    //
////////////////////////////////////////

void calCuTreeRadius(const std::vector<float> &r,std::vector<float> &tree_r,int layer){
    float sc1 = 1,sc2 = 1,sc3 = 1,sc4 = 1,sc5 = 1,sc6 = 1;

    switch(layer){
        case 3:{
            if(tree_r.size() != 3) {
                std::cout<<"Tree Radius Error!"<<std::endl;
                break;
            }
            tree_r[0] = r[0] * r[0];
            tree_r[1] = (r[1]+0.5*r[1]) * (r[1]+0.5*r[1]);
            tree_r[2] = r[2] * r[2];

            };break;
        case 4:{
            if(tree_r.size() != 4) {
                std::cout<<"Tree Radius Error!"<<std::endl;
                break;
            }
            tree_r[0] = r[0] * r[0];
            tree_r[1] = (r[1]+r[2]+0.5*r[2]) * (r[1]+r[2]+0.5*r[2]);
            tree_r[2] = (r[2]+0.5*r[2]) * (r[2]+0.5*r[2]);
            tree_r[3] = (r[3]) * (r[3]);
            };break;

        case 5:{
            if(tree_r.size() != 5) {
                std::cout<<"Tree Radius Error!"<<std::endl;
                break;
            }
            tree_r[0] = r[0] * r[0];
            tree_r[1] = (r[1]+r[2]+r[3]+0.5*r[3]) * (r[1]+r[2]+r[3]+0.5*r[3]);
            tree_r[2] = (r[2]+r[3]+0.5*r[3]) * (r[2]+r[3]+0.5*r[3]);
            tree_r[3] = (r[3]+0.5*r[3]) * (r[3]+0.5*r[3]);
            tree_r[4] = r[4] * r[4];



        };break;

        case 6:{
            if(tree_r.size() != 6) {
                std::cout<<"Tree Radius Error!"<<std::endl;
                break;
            }
            tree_r[0] =  r[0]*r[0];
            tree_r[1] = (r[1]+r[2]+r[3]+r[4]+0.5*r[4]) * (r[1]+r[2]+r[3]+r[4]+0.5*r[4]);
            tree_r[2] = (r[2]+r[3]+r[4]+0.5*r[4]) * (r[2]+r[3]+r[4]+0.5*r[4]);
            tree_r[3] = (r[3]+r[4]+0.5*r[4]) * (r[3]+r[4]+0.5*r[4]);
            tree_r[4] = (r[4]+0.5*r[4]) * (r[4]+0.5*r[4]);
            tree_r[5] = r[5] * r[5];

   
        };break;

        case 7:{
            if(tree_r.size() != 7) {
                std::cout<<"Tree Radius Error!"<<std::endl;
                break;
            }
            tree_r[0] = r[0] * r[0];
            tree_r[1] = (r[1]+r[2]+r[3]+r[4]+r[5]+0.5*r[5]) * (r[1]+r[2]+r[3]+r[4]+r[5]+0.5*r[5]);
            tree_r[2] = (r[2]+r[3]+r[4]+r[5]+0.5*r[5]) * (r[2]+r[3]+r[4]+r[5]+0.5*r[5]);
            tree_r[3] = (r[3]+r[4]+r[5]+0.5*r[5]) * (r[3]+r[4]+r[5]+0.5*r[5]);
            tree_r[4] = (r[4]+r[5]+0.5*r[5]) * (r[4]+r[5] +0.5*r[5]);
            tree_r[5] = (r[5]+0.5*r[5]) *(r[5]+0.5*r[5]);
            tree_r[6] = r[6] * r[6];
        };break;

    }
}

void calCuTreeRadius(const torch::Tensor &r, torch::Tensor &tree_r,int layer){
    float sc1 = 1,sc2 = 1,sc3 = 1,sc4 = 1,sc5 = 1,sc6 = 1;

    switch(layer){
        case 3:{
            if(tree_r.size(0) != 3) {
                std::cout<<"Tree Radius Error!"<<std::endl;
                break;
            }
            tree_r[0] = r[0] * r[0];
            tree_r[1] = (r[1]+0.5*r[1]) * (r[1]+0.5*r[1]);
            tree_r[2] = r[2] * r[2];
        };break;
        
        case 4:{   
            if(tree_r.size(0) != 4) {
                std::cout<<"Tree Radius Error!"<<std::endl;
                break;
            }
            tree_r[0] = r[0] * r[0];
            tree_r[1] = (r[1]+r[2]+0.5*r[2]) * (r[1]+r[2]+0.5*r[2]);
            tree_r[2] = (r[2]+0.5*r[2]) * (r[2]+0.5*r[2]);
            tree_r[3] = r[3] * r[3];

            };break;

        case 5:{
            if(tree_r.size(0) != 5) {
                std::cout<<"Tree Radius Error!"<<std::endl;
                break;
            }

            tree_r[0] = r[0] * r[0];
            tree_r[1] = (r[1]+r[2]+r[3]+0.5*r[3]) * (r[1]+r[2]+r[3]+0.5*r[3]);
            tree_r[2] = (r[2]+r[3]+0.5*r[3]) * (r[2]+r[3]+0.5*r[3]);
            tree_r[3] = (r[3]+0.5*r[3]) * (r[3]+0.5*r[3]);
            tree_r[4] = r[4] * r[4];

        };break;

        case 6:{
            if(tree_r.size(0) != 6) {
                std::cout<<"Tree Radius Error!"<<std::endl;
                break;
            }

            tree_r[0] =  r[0]*r[0];
            tree_r[1] = (r[1]+r[2]+r[3]+r[4]+0.5*r[4]) * (r[1]+r[2]+r[3]+r[4]+0.5*r[4]);
            tree_r[2] = (r[2]+r[3]+r[4]+0.5*r[4]) * (r[2]+r[3]+r[4]+0.5*r[4]);
            tree_r[3] = (r[3]+r[4]+0.5*r[4]) * (r[3]+r[4]+0.5*r[4]);
            tree_r[4] = (r[4]+0.5*r[4]) * (r[4]+0.5*r[4]);
            tree_r[5] = r[5] * r[5];

        };break;

        case 7:{
            if(tree_r.size(0) != 7) {
                std::cout<<"Tree Radius Error!"<<std::endl;
                break;
            }
            tree_r[0] = r[0] * r[0];
            tree_r[1] = (r[1]+r[2]+r[3]+r[4]+r[5]+0.5*r[5]) * (r[1]+r[2]+r[3]+r[4]+r[5]+0.5*r[5]);
            tree_r[2] = (r[2]+r[3]+r[4]+r[5]+0.5*r[5]) * (r[2]+r[3]+r[4]+r[5]+0.5*r[5]);
            tree_r[3] = (r[3]+r[4]+r[5]+0.5*r[5]) * (r[3]+r[4]+r[5]+0.5*r[5]);
            tree_r[4] = (r[4]+r[5]+0.5*r[5]) * (r[4]+r[5] +0.5*r[5]);
            tree_r[5] = (r[5]+0.5*r[5]) *(r[5]+0.5*r[5]);
            tree_r[6] = r[6] * r[6];
        };break;

    }
}

__global__ void isInShell_JumpingSphere_8_3l(const float *query_pt,
							 const float *IFPS_SphereTree,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r, 
                             int *res_gpu,
                             int qsize)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= qsize) return;

    int tmp_idx=idx*3;
    //justify if the point is in the first layer space
    if (dist(&IFPS_SphereTree[0], &query_pt[tmp_idx]) > r[0])
    {
        return;
    }

    for(int i = 1;i<=8;i++)
    {
        int tmp_i=i*3;
        if(dist(&IFPS_SphereTree[tmp_i], &query_pt[tmp_idx]) <= r[1])
        {
            for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
            {
                int tmp_j=j*3;
                if(dist(&IFPS_SphereTree[tmp_j], &query_pt[tmp_idx]) <= r[2])
                {
                    res_gpu[idx] = 1;
                    return;
                }
            }
        }
    }
    
}


__global__ void isInShell_JumpingSphere_8_4l(const Eigen::Vector3f *query_pt,
							 const Eigen::Vector3f *IFPS_SphereTree,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r,
                             int *res_gpu,
                             int qsize)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= qsize) return;


    //justify if the point is in the first layer space
	if (dist(IFPS_SphereTree[0], query_pt[idx]) > r[0])
	{
        res_gpu[idx] = 0;
		return;
	}

    for(int i = 1;i<=8;i++)
    {
        if(dist(IFPS_SphereTree[i], query_pt[idx]) <= r[1])
        {
            for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
            {
                if(dist(IFPS_SphereTree[j], query_pt[idx]) <= r[2])
                {
                    for(int k=ChildFirstIndex[j];k < ChildFirstIndex[j] + ChildNum[j]; k++)
                    {
                        if(dist(IFPS_SphereTree[k], query_pt[idx]) <= r[3])
                        {
							res_gpu[idx] = 1;
							return;
						}
                    }
                }
            }
        }
    }

}

__global__ void isInShell_JumpingSphere_8_4l(const float *query_pt,
							 const float *IFPS_SphereTree,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r, 
                             int *res_gpu,
                             int qsize)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= qsize) return;

    int tmp_idx=idx*3;

    //justify if the point is in the first layer space
    if (dist(&IFPS_SphereTree[0], &query_pt[tmp_idx]) > r[0])
    {
        return;
    }

    for(int i = 1;i<=8;i++)
    {
        int tmp_i=i*3;
        if(dist(&IFPS_SphereTree[tmp_i], &query_pt[tmp_idx]) <= r[1])
        {
            for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
            {
                int tmp_j=j*3;
                if(dist(&IFPS_SphereTree[tmp_j], &query_pt[tmp_idx]) <= r[2])
                {
                    for(int k=ChildFirstIndex[j];k < ChildFirstIndex[j] + ChildNum[j]; k++)
                    {
                        int tmp_k=k*3;
                        if(dist(&IFPS_SphereTree[tmp_k], &query_pt[tmp_idx]) <= r[3])
                        {
                            res_gpu[idx] = 1;
                            return;
                        }
                    }
                }
            }
        }
    }

}

__global__ void isInShell_JumpingSphere_8_4l(const float *query_pt,
							 const float *IFPS_SphereTree,
                             const int *IFPS_SPhereTree_idx,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r, 
                             int *res_gpu,
                             int *res_id_gpu,
                             int qsize)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= qsize) return;

    int tmp_idx=idx*3;

    //justify if the point is in the first layer space
    if (dist(&IFPS_SphereTree[0], &query_pt[tmp_idx]) > r[0])
    {
        return;
    }

    res_id_gpu[idx * 4] = 0;

    for(int i = 1;i<=8;i++)
    {
        int tmp_i=i*3;
        if(dist(&IFPS_SphereTree[tmp_i], &query_pt[tmp_idx]) <= r[1])
        {
            res_id_gpu[idx * 4 + 1] = IFPS_SPhereTree_idx[i];

            for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
            {
                int tmp_j=j*3;
                if(dist(&IFPS_SphereTree[tmp_j], &query_pt[tmp_idx]) <= r[2])
                {
                    res_id_gpu[idx * 4 + 2] = IFPS_SPhereTree_idx[j];
                    for(int k=ChildFirstIndex[j];k < ChildFirstIndex[j] + ChildNum[j]; k++)
                    {
                        int tmp_k=k*3;
                        if(dist(&IFPS_SphereTree[tmp_k], &query_pt[tmp_idx]) <= r[3])
                        {
                            res_id_gpu[idx * 4 + 3] = IFPS_SPhereTree_idx[k];
                            res_gpu[idx] = 1;
                            return;
                        }
                    }
                }
            }
        }
    }

}


__global__ void isInShell_JumpingSphere_8_5l(const Eigen::Vector3f *query_pt,
							 const Eigen::Vector3f *IFPS_SphereTree,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r,
                             int *res_gpu,
                             int qsize)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= qsize) return;

    //justify if the point is in the first layer space
	if (dist(IFPS_SphereTree[0], query_pt[idx]) > r[0])
	{
        res_gpu[idx] = 0;
		return;
	}

    for(int i = 1;i<=8;i++)
    {
        if(dist(IFPS_SphereTree[i], query_pt[idx]) <= r[1])
        {
            for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
            {
                if(dist(IFPS_SphereTree[j], query_pt[idx]) <= r[2])
                {
                    for(int k=ChildFirstIndex[j];k < ChildFirstIndex[j] + ChildNum[j]; k++)
                    {
                        if(dist(IFPS_SphereTree[k], query_pt[idx]) <= r[3])
                        {
							for(int m=ChildFirstIndex[k];m < ChildFirstIndex[k] + ChildNum[k]; m++)
							{
								if(dist(IFPS_SphereTree[m], query_pt[idx]) <= r[4])
								{
									res_gpu[idx] = 1;
									return;
								}
							}
						}
                    }
                }
            }
        }
    }

    res_gpu[idx] = 0;
    return;
}



__global__ void isInShell_JumpingSphere_8_5l(const float *query_pt,
							 const float *IFPS_SphereTree,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r,
                             int *res_gpu,
                             int qsize)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= qsize) return;



    //justify if the point is in the first layer space
    int tmp_idx=idx*3;
    if (dist(&IFPS_SphereTree[0], &query_pt[tmp_idx]) > r[0])
    {
        return;
    }

    for(int i = 1;i<=8;i++)
    {
        int tmp_i=i*3;
        if(dist(&IFPS_SphereTree[tmp_i], &query_pt[tmp_idx]) <= r[1])
        {
            for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
            {
                int tmp_j=j*3;
                if(dist(&IFPS_SphereTree[tmp_j], &query_pt[tmp_idx]) <= r[2])
                {
                    for(int k=ChildFirstIndex[j];k < ChildFirstIndex[j] + ChildNum[j]; k++)
                    {
                        int tmp_k=k*3;
                        if(dist(&IFPS_SphereTree[tmp_k], &query_pt[tmp_idx]) <= r[3])
                        {
                            for(int m=ChildFirstIndex[k];m < ChildFirstIndex[k] + ChildNum[k]; m++)
                            {
                                int tmp_m=m*3;
                                if(dist(&IFPS_SphereTree[tmp_m], &query_pt[tmp_idx]) <= r[4])
                                {
                                    res_gpu[idx] = 1;
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
}


__global__ void isInShell_JumpingSphere_8_6l(const Eigen::Vector3f *query_pt,
							 const Eigen::Vector3f *IFPS_SphereTree,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r,
                             int *res_gpu,
                             int qsize)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= qsize) return;

    //justify if the point is in the first layer space
	if (dist(IFPS_SphereTree[0], query_pt[idx]) > r[0])
	{
        res_gpu[idx] = 0;
		return;
	}

    for(int i = 1;i<=8;i++)
    {
        if(dist(IFPS_SphereTree[i], query_pt[idx]) <= r[1])
        {
            for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
            {
                if(dist(IFPS_SphereTree[j], query_pt[idx]) <= r[2])
                {
                    for(int k=ChildFirstIndex[j];k < ChildFirstIndex[j] + ChildNum[j]; k++)
                    {
                        if(dist(IFPS_SphereTree[k], query_pt[idx]) <= r[3])
                        {
							for(int m=ChildFirstIndex[k];m < ChildFirstIndex[k] + ChildNum[k]; m++)
							{
								if(dist(IFPS_SphereTree[m], query_pt[idx]) <= r[4])
								{
									for(int n=ChildFirstIndex[m];n < ChildFirstIndex[m] + ChildNum[m]; n++)
									{
										if(dist(IFPS_SphereTree[n], query_pt[idx]) <= r[5])
										{
											res_gpu[idx] = 1;
											return;
										}
									}
								}
							}
						}
                    }
                }
            }
        }
    }
    res_gpu[idx] = 0;
    return;
}

__global__ void isInShell_JumpingSphere_8_6l(const float *query_pt,
							 const float *IFPS_SphereTree,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r,
                             int *res_gpu,
                             int qsize)
{
   uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
   if(idx >= qsize) return;


    int tmp_idx=idx*3;
    if (dist(&IFPS_SphereTree[0], &query_pt[tmp_idx]) > r[0])
    {
        return;
    }

    for(int i = 1;i<=8;i++)
    {
        int tmp_i=i*3;
        if(dist(&IFPS_SphereTree[tmp_i], &query_pt[tmp_idx]) <= r[1])
        {
            for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
            {
                int tmp_j=j*3;
                if(dist(&IFPS_SphereTree[tmp_j], &query_pt[tmp_idx]) <= r[2])
                {
                    for(int k=ChildFirstIndex[j];k < ChildFirstIndex[j] + ChildNum[j]; k++)
                    {
                        int tmp_k=k*3;
                        if(dist(&IFPS_SphereTree[tmp_k], &query_pt[tmp_idx]) <= r[3])
                        {
                            for(int m=ChildFirstIndex[k];m < ChildFirstIndex[k] + ChildNum[k]; m++)
                            {
                                int tmp_m=m*3;
                                if(dist(&IFPS_SphereTree[tmp_m], &query_pt[tmp_idx]) <= r[4])
                                {
                                    for(int n=ChildFirstIndex[m];n < ChildFirstIndex[m] + ChildNum[m]; n++)
                                    {
                                        int tmp_n=n*3;
                                        if(dist(&IFPS_SphereTree[tmp_n], &query_pt[tmp_idx]) <= r[5])
                                        {
                                            res_gpu[idx] = 1;
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    
}



__global__ void isInShell_JumpingSphere_8_7l(const Eigen::Vector3f *query_pt,
							 const Eigen::Vector3f *IFPS_SphereTree,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r,
                             int *res_gpu,
                             int qsize)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= qsize) return;
    
	if (dist(IFPS_SphereTree[0], query_pt[idx]) > r[0])
	{
        res_gpu[idx] = 0;
		return;
	}

    for(int i = 1;i<=8;i++)
    {
        if(dist(IFPS_SphereTree[i], query_pt[idx]) <= r[1])
        {
            for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
            {
                if(dist(IFPS_SphereTree[j], query_pt[idx]) <= r[2])
                {
                    for(int k=ChildFirstIndex[j];k < ChildFirstIndex[j] + ChildNum[j]; k++)
                    {
                        if(dist(IFPS_SphereTree[k], query_pt[idx]) <= r[3])
                        {
                            for(int m=ChildFirstIndex[k];m < ChildFirstIndex[k] + ChildNum[k]; m++)
                            {
                                if(dist(IFPS_SphereTree[m], query_pt[idx]) <= r[4])
                                {
                                    for(int n=ChildFirstIndex[m];n < ChildFirstIndex[m] + ChildNum[m]; n++)
                                    {
                                        if(dist(IFPS_SphereTree[n], query_pt[idx]) <= r[5])
                                        {
											for(int p=ChildFirstIndex[n];p < ChildFirstIndex[n] + ChildNum[n]; p++)
											{
												if(dist(IFPS_SphereTree[p], query_pt[idx]) <= r[6])
												{
													res_gpu[idx] = 1;
													return;
												}
											}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    res_gpu[idx] = 0;
    return;
}

__global__ void isInShell_JumpingSphere_8_7l(const float *query_pt,
							 const float *IFPS_SphereTree,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r,
                             int *res_gpu,
                             int qsize)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= qsize) return;


    int tmp_idx=idx*3;
    //justify if the point is in the first layer space
    if (dist(&IFPS_SphereTree[0], &query_pt[tmp_idx]) > r[0])
    {
        return;
    }

    for(int i = 1;i<=8;i++)
    {
        int tmp_i=i*3;
        if(dist(&IFPS_SphereTree[tmp_i], &query_pt[tmp_idx]) <= r[1])
        {
            for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
            {
                int tmp_j=j*3;
                if(dist(&IFPS_SphereTree[tmp_j], &query_pt[tmp_idx]) <= r[2])
                {
                    for(int k=ChildFirstIndex[j];k < ChildFirstIndex[j] + ChildNum[j]; k++)
                    {
                        int tmp_k=k*3;
                        if(dist(&IFPS_SphereTree[tmp_k], &query_pt[tmp_idx]) <= r[3])
                        {
                            for(int m=ChildFirstIndex[k];m < ChildFirstIndex[k] + ChildNum[k]; m++)
                            {
                                int tmp_m=m*3;
                                if(dist(&IFPS_SphereTree[tmp_m], &query_pt[tmp_idx]) <= r[4])
                                {
                                    for(int n=ChildFirstIndex[m];n < ChildFirstIndex[m] + ChildNum[m]; n++)
                                    {
                                        int tmp_n=n*3;
                                        if(dist(&IFPS_SphereTree[tmp_n], &query_pt[tmp_idx]) <= r[5])
                                        {
                                            for(int p=ChildFirstIndex[n];p < ChildFirstIndex[n] + ChildNum[n]; p++)
                                            {
                                                int tmp_p=p*3;
                                                if(dist(&IFPS_SphereTree[tmp_p], &query_pt[tmp_idx]) <= r[6])
                                                {
                                                    res_gpu[idx] = 1;
                                                    return;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
}



inline void isInShell_JumpingSphere_8_3l_cpu(const float *query_pt,
							 const float *IFPS_SphereTree,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r, 
                             int *res_gpu,
                             int qsize)
{
    for(int idx = 0 ;idx<qsize;idx++){
        int tmp_idx=idx*3;
        float r0_2=r[0];
        //justify if the point is in the first layer space
        if (dist(&IFPS_SphereTree[0], &query_pt[tmp_idx]) > r0_2)
        {
            continue;
        }

        for(int i = 1;i<=8;i++)
        {
            int tmp_i=i*3;
            if(dist(&IFPS_SphereTree[tmp_i], &query_pt[tmp_idx]) <= r[1])
            {
                for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
                {
                    int tmp_j=j*3;
                    if(dist(&IFPS_SphereTree[tmp_j], &query_pt[tmp_idx]) <= r[2])
                    {
                        res_gpu[idx] = 1;
                        //revised by Dawnaye 2025 2 27
                        //continue;
                        break;
                    }
                    
                }
                if(res_gpu[idx]==1) break;
            }
        }
    }

}


inline void isInShell_JumpingSphere_8_4l_cpu(const float *query_pt,
							 const float *IFPS_SphereTree,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r, 
                             int *res_gpu,
                             int qsize)
{
    for(int idx = 0 ;idx<qsize;idx++){
        int tmp_idx=idx*3;
        float r0_2=r[0];
        //justify if the point is in the first layer space
        if (dist(&IFPS_SphereTree[0], &query_pt[tmp_idx]) > r0_2)
        {
            continue;
        }

        for(int i = 1;i<=8;i++)
        {
            int tmp_i=i*3;
            if(dist(&IFPS_SphereTree[tmp_i], &query_pt[tmp_idx]) <= r[1])
            {
                for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
                {
                    int tmp_j=j*3;
                    if(dist(&IFPS_SphereTree[tmp_j], &query_pt[tmp_idx]) <= r[2])
                    {
                        for(int k=ChildFirstIndex[j];k < ChildFirstIndex[j] + ChildNum[j]; k++)
                        {
                            int tmp_k=k*3;
                            if(dist(&IFPS_SphereTree[tmp_k], &query_pt[tmp_idx]) <= r[3])
                            {
                                res_gpu[idx] = 1;
                                //continue;
                                break;
                            }
                        }
                        if(res_gpu[idx]==1) break;

                    }
                }                
                if(res_gpu[idx]==1) break;
           }
        }
    }

}


inline void isInShell_JumpingSphere_8_5l_cpu(const float *query_pt,
							 const float *IFPS_SphereTree,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r,
                             int *res_gpu,
                             int qsize)
{
    
    for(int idx = 0;idx <qsize;idx ++ ){
        //justify if the point is in the first layer space
        int tmp_idx=idx*3;

        if (dist(&IFPS_SphereTree[0], &query_pt[tmp_idx]) > r[0])
        {
            
            continue;
        }

        for(int i = 1;i<=8;i++)
        {
            int tmp_i=i*3;
            if(dist(&IFPS_SphereTree[tmp_i], &query_pt[tmp_idx]) <= r[1])
            {
                for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
                {
                    int tmp_j=j*3;
                    if(dist(&IFPS_SphereTree[tmp_j], &query_pt[tmp_idx]) <= r[2])
                    {
                            // res_gpu[idx] = 1;
                            // continue;
                        for(int k=ChildFirstIndex[j];k < ChildFirstIndex[j] + ChildNum[j]; k++)
                        {

                            int tmp_k=k*3;
                            if(dist(&IFPS_SphereTree[tmp_k], &query_pt[tmp_idx]) <= r[3])
                            {
                                for(int m=ChildFirstIndex[k];m < ChildFirstIndex[k] + ChildNum[k]; m++)
                                {
                                    int tmp_m=m*3;
                                    if(dist(&IFPS_SphereTree[tmp_m], &query_pt[tmp_idx]) <= r[4])
                                    {
                                        res_gpu[idx] = 1;
                                        //continue;
                                        break;
                                    }
                                }                
                                if(res_gpu[idx]==1) break;

                            }
                        }
                        if(res_gpu[idx]==1) break;
                    }
                }
                if(res_gpu[idx]==1) break;
            }
        }
    }
    

}

inline void isInShell_JumpingSphere_8_6l_cpu(const float *query_pt,
							 const float *IFPS_SphereTree,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r,
                             int *res_gpu,
                             int qsize)
{

    for(int idx = 0 ;idx < qsize ; idx++){
        int tmp_idx=idx*3;
        float r0_2=r[0];
        //justify if the point is in the first layer space
        if (dist(&IFPS_SphereTree[0], &query_pt[tmp_idx]) > r0_2)
        {
            continue;
        }

        for(int i = 1;i<=8;i++)
        {
            int tmp_i=i*3;
            if(dist(&IFPS_SphereTree[tmp_i], &query_pt[tmp_idx]) <= r[1])
            {
                for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
                {
                    int tmp_j=j*3;
                    if(dist(&IFPS_SphereTree[tmp_j], &query_pt[tmp_idx]) <= r[2])
                    {
                        for(int k=ChildFirstIndex[j];k < ChildFirstIndex[j] + ChildNum[j]; k++)
                        {
                            int tmp_k=k*3;
                            if(dist(&IFPS_SphereTree[tmp_k], &query_pt[tmp_idx]) <= r[3])
                            {
                                for(int m=ChildFirstIndex[k];m < ChildFirstIndex[k] + ChildNum[k]; m++)
                                {
                                    int tmp_m=m*3;
                                    if(dist(&IFPS_SphereTree[tmp_m], &query_pt[tmp_idx]) <= r[4])
                                    {
                                        for(int n=ChildFirstIndex[m];n < ChildFirstIndex[m] + ChildNum[m]; n++)
                                        {
                                            int tmp_n=n*3;
                                            if(dist(&IFPS_SphereTree[tmp_n], &query_pt[tmp_idx]) <= r[5])
                                            {
                                                res_gpu[idx] = 1;
                                                //continue;
                                                break;
                                            }
                                        }
                                        if(res_gpu[idx]==1) break;
                                    }
                                }
                                if(res_gpu[idx]==1) break;
                            }
                        }
                        if(res_gpu[idx]==1) break;
                    }
                }
                if(res_gpu[idx]==1) break;
            }
        }
    }

}

inline void isInShell_JumpingSphere_8_7l_cpu(const float *query_pt,
							 const float *IFPS_SphereTree,
							 const int *ChildFirstIndex,
							 const int *ChildNum,
							 const float *r,
                             int *res_gpu,
                             int qsize)
{
    for(int idx = 0 ;idx < qsize ; idx ++ ){
        int tmp_idx=idx*3;
        float r0_2=r[0];
    //justify if the point is in the first layer space
        if (dist(&IFPS_SphereTree[0], &query_pt[tmp_idx]) > r0_2)
        {
            continue ;
        }

        for(int i = 1;i<=8;i++)
        {
            int tmp_i=i*3;
            if(dist(&IFPS_SphereTree[tmp_i], &query_pt[tmp_idx]) <= r[1])
            {
                for(int j=ChildFirstIndex[i];j < ChildFirstIndex[i] + ChildNum[i]; j++)
                {
                    int tmp_j=j*3;
                    if(dist(&IFPS_SphereTree[tmp_j], &query_pt[tmp_idx]) <= r[2])
                    {
                        for(int k=ChildFirstIndex[j];k < ChildFirstIndex[j] + ChildNum[j]; k++)
                        {
                            int tmp_k=k*3;
                            if(dist(&IFPS_SphereTree[tmp_k], &query_pt[tmp_idx]) <= r[3])
                            {
                                for(int m=ChildFirstIndex[k];m < ChildFirstIndex[k] + ChildNum[k]; m++)
                                {
                                    int tmp_m=m*3;
                                    if(dist(&IFPS_SphereTree[tmp_m], &query_pt[tmp_idx]) <= r[4])
                                    {
                                        for(int n=ChildFirstIndex[m];n < ChildFirstIndex[m] + ChildNum[m]; n++)
                                        {
                                            int tmp_n=n*3;
                                            if(dist(&IFPS_SphereTree[tmp_n], &query_pt[tmp_idx]) <= r[5])
                                            {
                                                for(int p=ChildFirstIndex[n];p < ChildFirstIndex[n] + ChildNum[n]; p++)
                                                {
                                                    int tmp_p=p*3;
                                                    if(dist(&IFPS_SphereTree[tmp_p], &query_pt[tmp_idx]) <= r[6])
                                                    {
                                                        res_gpu[idx] = 1;
                                                        //continue ;
                                                        break;
                                                    }
                                                }
                                                if(res_gpu[idx]==1) break;
                                            }
                                        }
                                        if(res_gpu[idx]==1) break;
                                    }
                                }
                                if(res_gpu[idx]==1) break;
                            }
                        }
                        if(res_gpu[idx]==1) break;
                    }
                }
                if(res_gpu[idx]==1) break;
            }
        }
    }   
}




std::pair<std::vector<int>,int> check_treepts_cpu_time(const std::vector<float> &vec_infer_pts,
                            const std::vector<float> &vec_IFPS_SphereTree,
							const std::vector<int> &vec_ChildFirstIndex,
							const std::vector<int> &vec_ChildNum,
							const std::vector<float> &vec_r,
                            const int layer){
    unsigned int qs = vec_infer_pts.size() / 3;
    std::vector<float> vec_tree_r(layer);
    std::vector<int> vec_res(qs);
    calCuTreeRadius(vec_r,vec_tree_r,layer);


    float *infer_pts=new float[vec_infer_pts.size()];
    float *IFPS_SphereTree=new float[vec_IFPS_SphereTree.size()];
    int *ChildFirstIndex=new int[vec_ChildFirstIndex.size()];
    int *ChildNum=new int[vec_ChildNum.size()];
    float *r=new float[vec_r.size()];
    float *tree_r=new float[vec_tree_r.size()];
    int *res=new int[vec_res.size()];

    for(int i=0;i<vec_infer_pts.size();i++){
        infer_pts[i]=vec_infer_pts[i];
    }
    for(int i=0;i<vec_IFPS_SphereTree.size();i++){
        IFPS_SphereTree[i]=vec_IFPS_SphereTree[i];
    }

    for(int i=0;i<vec_ChildFirstIndex.size();i++){
            ChildFirstIndex[i]=vec_ChildFirstIndex[i];
        }

    for(int i=0;i<vec_ChildNum.size();i++){
            ChildNum[i]=vec_ChildNum[i];
        }

    for(int i=0;i<vec_r.size();i++){
            r[i]=vec_r[i];
        }

    for(int i=0;i<vec_tree_r.size();i++){
            tree_r[i]=vec_tree_r[i];
        }

  

    double time = 0.0;

    switch(layer){
        case 3:{

            std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
            isInShell_JumpingSphere_8_3l_cpu(infer_pts,IFPS_SphereTree,ChildFirstIndex,ChildNum,tree_r,res,qs);
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
        }
        
        ;break;
        case 4:{

            std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
            isInShell_JumpingSphere_8_4l_cpu(infer_pts,IFPS_SphereTree,ChildFirstIndex,ChildNum,tree_r,res,qs);
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
        }
        
        ;break;

        case 5:{
            std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
            isInShell_JumpingSphere_8_5l_cpu(infer_pts,IFPS_SphereTree,ChildFirstIndex,ChildNum,tree_r,res,qs);
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
        };break;
        
        case 6:{

            std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
            isInShell_JumpingSphere_8_6l_cpu(infer_pts,IFPS_SphereTree,ChildFirstIndex,ChildNum,tree_r,res,qs);
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
        };break;

        case 7:{

            std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
            isInShell_JumpingSphere_8_7l_cpu(infer_pts,IFPS_SphereTree,ChildFirstIndex,ChildNum,tree_r,res,qs);
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
        };break;
    }

      for(int i=0;i<vec_res.size();i++){
            vec_res[i]=res[i];
        }
    
    return std::make_pair(vec_res,time);

}


std::vector<int> check_treepts_gpu(const std::vector<Eigen::Vector3f> &infer_pts,
                            const std::vector<Eigen::Vector3f> &IFPS_SphereTree,
							const std::vector<int> &ChildFirstIndex,
							const std::vector<int> &ChildNum,
							const std::vector<float> &r,
                            const int layer){
    unsigned int threads = 256;
    unsigned int blocks = (infer_pts.size() + threads -1) / threads;

    std::vector<float> tree_r(layer);
    calCuTreeRadius(r,tree_r,layer);

    GpuArray<Eigen::Vector3f> infer_pts_gpu(infer_pts,infer_pts.size());
    GpuArray<Eigen::Vector3f> ifps_st_gpu(IFPS_SphereTree,IFPS_SphereTree.size());
    GpuArray<int> child_first_index_gpu(ChildFirstIndex,ChildFirstIndex.size());
    GpuArray<int> child_num_gpu(ChildNum,ChildNum.size());
    GpuArray<float> r_gpu(r,r.size());
    GpuArray<float> tree_r_gpu(tree_r,tree_r.size());
    std::shared_ptr<Timer> timer = std::make_shared<Timer>();


    std::vector<int> res(infer_pts.size());
    GpuArray<int> res_gpu(res,res.size());
    unsigned int qs = infer_pts.size();

    switch(layer){
        case 4:{
            timer->start();

            isInShell_JumpingSphere_8_4l<<<blocks,threads>>>(infer_pts_gpu.ptr(),ifps_st_gpu.ptr(),child_first_index_gpu.ptr(),child_num_gpu.ptr(),tree_r_gpu.ptr(),res_gpu.ptr(),qs);
            
            timer->stop();
            timer->printTime();
        };break;

        case 5:{
            timer->start();

            isInShell_JumpingSphere_8_5l<<<blocks,threads>>>(infer_pts_gpu.ptr(),ifps_st_gpu.ptr(),child_first_index_gpu.ptr(),child_num_gpu.ptr(),tree_r_gpu.ptr(),res_gpu.ptr(),qs);       
            
            timer->stop();
            timer->printTime();
        };break;
        
        case 6:{
            timer->start();

            isInShell_JumpingSphere_8_6l<<<blocks,threads>>>(infer_pts_gpu.ptr(),ifps_st_gpu.ptr(),child_first_index_gpu.ptr(),child_num_gpu.ptr(),tree_r_gpu.ptr(),res_gpu.ptr(),qs);         
            
            timer->stop();
            timer->printTime();
        };break;

        case 7:{
            timer->start();

            isInShell_JumpingSphere_8_7l<<<blocks,threads>>>(infer_pts_gpu.ptr(),ifps_st_gpu.ptr(),child_first_index_gpu.ptr(),child_num_gpu.ptr(),tree_r_gpu.ptr(),res_gpu.ptr(),qs);         
            
            timer->stop();
            timer->printTime();
        };break;
    }

    res_gpu.copyToCPU(res.data(),res.size());


    return res;

}

std::pair<torch::Tensor,float>  check_treepts_gpu_time(const torch::Tensor &infer_pts,
                            const torch::Tensor &IFPS_SphereTree,
							const torch::Tensor &ChildFirstIndex,
							const torch::Tensor &ChildNum,
							const torch::Tensor &r,
                            const int layer){
	unsigned int qs = infer_pts.size(0);
    unsigned int threads = 256;
    unsigned int blocks = (qs + threads-1) / threads;

    std::shared_ptr<Timer> timer = std::make_shared<Timer>();
    torch::Tensor tree_r = torch::zeros({layer},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor res_gpu = torch::zeros({qs}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));


    calCuTreeRadius(r,tree_r,layer);    

    switch(layer){
        case 3:{
            timer->start();
            isInShell_JumpingSphere_8_3l<<<blocks,threads>>>(infer_pts.data_ptr<float>(),IFPS_SphereTree.data_ptr<float>(),ChildFirstIndex.data_ptr<int>(),ChildNum.data_ptr<int>(),tree_r.data_ptr<float>(),res_gpu.data_ptr<int>(),qs);
            
            timer->stop();
        }
        
        ;break;
        case 4:{

            timer->start();
            isInShell_JumpingSphere_8_4l<<<blocks,threads>>>(infer_pts.data_ptr<float>(),IFPS_SphereTree.data_ptr<float>(),ChildFirstIndex.data_ptr<int>(),ChildNum.data_ptr<int>(),tree_r.data_ptr<float>(),res_gpu.data_ptr<int>(),qs);
            
            timer->stop();
        }
        
        ;break;

        case 5:{
            timer->start();

            isInShell_JumpingSphere_8_5l<<<blocks,threads>>>(infer_pts.data_ptr<float>(),IFPS_SphereTree.data_ptr<float>(),ChildFirstIndex.data_ptr<int>(),ChildNum.data_ptr<int>(),tree_r.data_ptr<float>(),res_gpu.data_ptr<int>(),qs);
            
            timer->stop();
        };break;
        
        case 6:{

            timer->start();

            isInShell_JumpingSphere_8_6l<<<blocks,threads>>>(infer_pts.data_ptr<float>(),IFPS_SphereTree.data_ptr<float>(),ChildFirstIndex.data_ptr<int>(),ChildNum.data_ptr<int>(),tree_r.data_ptr<float>(),res_gpu.data_ptr<int>(),qs);
            
            timer->stop();
        };break;

        case 7:{
            timer->start();

            isInShell_JumpingSphere_8_7l<<<blocks,threads>>>(infer_pts.data_ptr<float>(),IFPS_SphereTree.data_ptr<float>(),ChildFirstIndex.data_ptr<int>(),ChildNum.data_ptr<int>(),tree_r.data_ptr<float>(),res_gpu.data_ptr<int>(),qs);
            
            timer->stop();
        };break;
    }
    
    return std::make_pair(res_gpu,timer->time);

}

std::tuple<torch::Tensor,torch::Tensor,float>  check_treepts_gpu_id(const torch::Tensor &infer_pts,
                            const torch::Tensor &IFPS_SphereTree,
                            const torch::Tensor &IFPS_SPhereTree_idx,
							const torch::Tensor &ChildFirstIndex,
							const torch::Tensor &ChildNum,
							const torch::Tensor &r,
                            const int layer){
	unsigned int qs = infer_pts.size(0);
    unsigned int threads = 256;
    unsigned int blocks = (qs + threads-1) / threads;

    std::shared_ptr<Timer> timer = std::make_shared<Timer>();
    torch::Tensor tree_r = torch::zeros({layer},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor res_gpu = torch::zeros({qs}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    torch::Tensor res_id_gpu = torch::zeros({qs * 4}, torch::TensorOptions().dtype(torch::kInt32)).fill_(-1).to(torch::kCUDA);
    calCuTreeRadius(r,tree_r,layer);    

    switch(layer){
        case 3:{
            res_id_gpu = torch::zeros({qs * 3}, torch::TensorOptions().dtype(torch::kInt32)).fill_(-1).to(torch::kCUDA);

            timer->start();
            isInShell_JumpingSphere_8_3l<<<blocks,threads>>>(infer_pts.data_ptr<float>(),IFPS_SphereTree.data_ptr<float>(),ChildFirstIndex.data_ptr<int>(),ChildNum.data_ptr<int>(),tree_r.data_ptr<float>(),res_gpu.data_ptr<int>(),qs);
            
            timer->stop();
        }
        
        ;break;
        case 4:{
            res_id_gpu = torch::zeros({qs * 4}, torch::TensorOptions().dtype(torch::kInt32)).fill_(-1).to(torch::kCUDA);

            timer->start();
            isInShell_JumpingSphere_8_4l<<<blocks,threads>>>(infer_pts.data_ptr<float>(),IFPS_SphereTree.data_ptr<float>(),IFPS_SPhereTree_idx.data_ptr<int>(),ChildFirstIndex.data_ptr<int>(),ChildNum.data_ptr<int>(),tree_r.data_ptr<float>(),res_gpu.data_ptr<int>(),res_id_gpu.data_ptr<int>(),qs);
            
            timer->stop();
        }
        
        ;break;

        case 5:{
            res_id_gpu = torch::zeros({qs * 5}, torch::TensorOptions().dtype(torch::kInt32)).fill_(-1).to(torch::kCUDA);
            timer->start();

            isInShell_JumpingSphere_8_5l<<<blocks,threads>>>(infer_pts.data_ptr<float>(),IFPS_SphereTree.data_ptr<float>(),ChildFirstIndex.data_ptr<int>(),ChildNum.data_ptr<int>(),tree_r.data_ptr<float>(),res_gpu.data_ptr<int>(),qs);
            
            timer->stop();
        };break;
        
        case 6:{
            res_id_gpu = torch::zeros({qs * 6}, torch::TensorOptions().dtype(torch::kInt32)).fill_(-1).to(torch::kCUDA);

            timer->start();

            isInShell_JumpingSphere_8_6l<<<blocks,threads>>>(infer_pts.data_ptr<float>(),IFPS_SphereTree.data_ptr<float>(),ChildFirstIndex.data_ptr<int>(),ChildNum.data_ptr<int>(),tree_r.data_ptr<float>(),res_gpu.data_ptr<int>(),qs);
            
            timer->stop();
        };break;

        case 7:{
            res_id_gpu = torch::zeros({qs * 7}, torch::TensorOptions().dtype(torch::kInt32)).fill_(-1).to(torch::kCUDA);

            timer->start();

            isInShell_JumpingSphere_8_7l<<<blocks,threads>>>(infer_pts.data_ptr<float>(),IFPS_SphereTree.data_ptr<float>(),ChildFirstIndex.data_ptr<int>(),ChildNum.data_ptr<int>(),tree_r.data_ptr<float>(),res_gpu.data_ptr<int>(),qs);
            
            timer->stop();
        };break;
    }
    
    return std::make_tuple(res_gpu,res_id_gpu,timer->time);

}
