#include "ifps.cuh"

// base function
__host__ __device__ float F2_distance(const std::vector<float> &p1,const std::vector<float> &p2){
    return (p1[0]-p2[0]) *(p1[0]-p2[0]) 
    + (p1[1]-p2[1]) *(p1[1]-p2[1]) 
    + (p1[2]-p2[2]) *(p1[2]-p2[2]);
}

__host__ __device__ float F1_distance(const std::vector<float> &p1,const std::vector<float> &p2){
    return std::abs(p1[0]-p2[0]) + std::abs(p1[1]-p2[1]) + std::abs(p1[2]-p2[2]);
}

__host__ __device__ float Fn_distance(const std::vector<float> &p1,const std::vector<float> &p2){
    return std::max(std::abs(p1[0]-p2[0]) , std::max(std::abs(p1[1]-p2[1]),std::abs(p1[2]-p2[2])));
}


__global__ void min_distance_cu(float *dist_cur_allPt_gpu,float *dist_cur_allPt_min_gpu){
    uint32_t i  = blockDim.x * blockIdx.x + threadIdx.x;
    if(dist_cur_allPt_gpu[i] < dist_cur_allPt_min_gpu[i]){
        dist_cur_allPt_min_gpu[i] = dist_cur_allPt_gpu[i];
    }
}



__global__  void inverse_shell_cu(
const Eigen::Vector3f *infer_pts_gpu,
const Eigen::Vector3f *ordered_pt_gpu,
const float *ordered_radi_gpu,
int *contourBound_cur_gpu,
uint32_t num,
uint32_t m,
uint32_t normChoice
){
    uint32_t i  = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= num) return;
    float min_dist = 1e8f;
    float dist_cur_prevSp_gpu = -1;

    // // #pragma unroll
    for(uint32_t kk = 0 ;kk < m;kk++){
        float x = infer_pts_gpu[i][0] - ordered_pt_gpu[kk][0];
        float y = infer_pts_gpu[i][1] - ordered_pt_gpu[kk][1];
        float z = infer_pts_gpu[i][2] - ordered_pt_gpu[kk][2];
        if(normChoice == 2){
            dist_cur_prevSp_gpu = x*x + y*y + z *z;
        }else if(normChoice == 1){
            dist_cur_prevSp_gpu = fabs(x) + fabs(y) + fabs(z);
        }else{
            dist_cur_prevSp_gpu = std::max(std::max(fabs(x),fabs(y)),fabs(z));
        }          
        
        if(dist_cur_prevSp_gpu < min_dist){
            min_dist = dist_cur_prevSp_gpu;
        }
        if (min_dist > ordered_radi_gpu[kk+1]){
            contourBound_cur_gpu[i] = 0;
            break;
        }
    }
}


__global__  void inverse_shell_cu(
const float *infer_pts_gpu,
const float *ordered_pt_gpu,
const float *ordered_radi_gpu,
int *contourBound_cur_gpu,
uint32_t num,uint32_t m,uint32_t normChoice
){
    uint32_t i  = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= num) return;
    float min_dist = 1e8f;
    float dist_cur_prevSp_gpu = -1;
    // // #pragma unroll

    for(uint32_t kk = 0 ;kk < m;kk++){
        float x = infer_pts_gpu[i*3] - ordered_pt_gpu[kk *3];
        float y = infer_pts_gpu[i*3 + 1] - ordered_pt_gpu[kk *3 + 1];
        float z = infer_pts_gpu[i*3 + 2] - ordered_pt_gpu[kk *3 + 2];
        if(normChoice == 2){
            dist_cur_prevSp_gpu = x*x + y*y + z *z;
        }else if(normChoice == 1){
            dist_cur_prevSp_gpu = fabs(x) + fabs(y) + fabs(z);
        }else{
            dist_cur_prevSp_gpu = std::max(std::max(fabs(x),fabs(y)),fabs(z));
        }                                                 
        
        if(dist_cur_prevSp_gpu < min_dist){
            min_dist = dist_cur_prevSp_gpu;
        }
        if (min_dist > ordered_radi_gpu[kk+1]){
            contourBound_cur_gpu[i] = 0;
            break;
        }
    }
}



/*
    fps
    -- fps function
    -- inverse_fps function
*/
std::vector<std::vector<float>> Ifps::fps(std::vector<std::vector<float>> &dense_pts,const unsigned int m,const unsigned int normChoice)
{
    unsigned int num_pts = dense_pts.size();
    std::vector<std::vector<float>> ordered_pt(m + 1,std::vector<float>(3,0.0));
    std::vector<float> dist_cur_allPt(num_pts,0);
    std::vector<float> dist_cur_allPt_min(num_pts,1 * 2.1);
    std::shared_ptr<Timer> timer = std::make_shared<Timer>();

    std::vector<float> cen_0 {0,0,0};
    std::vector<float> cur_pt = cen_0;

    ordered_pt[0] = cen_0;

    timer->start();
    if(normChoice == 1){
        funptr = std::make_shared<Funcptr>(&F1_distance);
    }else if(normChoice == 2){
        funptr = std::make_shared<Funcptr>(&F2_distance);
    }else if(normChoice == 100){
        funptr = std::make_shared<Funcptr>(&Fn_distance);
    }

    for(unsigned int i = 1;i < m + 1;i++){
        for(unsigned int j = 0;j < num_pts;j++){
            dist_cur_allPt[j] = (*funptr)(cur_pt,dense_pts[j]);
        }

        for(unsigned int j = 0; j < num_pts ;j++){
            if(dist_cur_allPt[j] < dist_cur_allPt_min[j]){
                dist_cur_allPt_min[j] = dist_cur_allPt[j];
            }
        }

        auto max_it = std::max_element(dist_cur_allPt_min.begin(),dist_cur_allPt_min.end());
        int ind = std::distance(dist_cur_allPt_min.begin(),max_it);
         
        cur_pt = dense_pts[ind];
        ordered_pt[i] = cur_pt;
    }

    timer->stop();
    timer->printTime();
    return ordered_pt;
}



std::vector<float> Ifps::inverse_fps(const std::vector<std::vector<float>> & ordered_pt,const unsigned int m,const unsigned int normChoice){
    std::shared_ptr<Timer> timer = std::make_shared<Timer>();
    std::vector<float> ordered_radi(m +1,0);

    if(normChoice == 2){
        funptr = std::make_shared<Funcptr>(&F2_distance);
    }else if(normChoice == 1){
        funptr = std::make_shared<Funcptr>(&F1_distance);
    }else{
        funptr = std::make_shared<Funcptr>(&Fn_distance);
    }

    std::cout<<"[INFO] INVERSE FPS"<<std::endl;
    timer->start();
    for(unsigned int i = 1;i < m + 1; i++){
        std::vector<float> dist_cur_allPt(i,0.0);
        for(unsigned int ii = 0; ii < i;ii++){
            dist_cur_allPt[ii] = (*funptr)(ordered_pt[ii],ordered_pt[i]);
        }
        ordered_radi[i] = *std::min_element(dist_cur_allPt.begin(),dist_cur_allPt.end()) * 1.05;

    }
    timer->stop();
    timer->printTime();
    return ordered_radi;
}


/*
    inverse shell
    -- the function to determines if points in shell
*/
std::vector<int> Ifps::inverse_shell_cpu(
    const std::vector<std::vector<float>> &infer_pts,
    const std::vector<std::vector<float>> &ordered_pt,
    const std::vector<float> &ordered_radi,
    const unsigned int m,
    const unsigned int normChoice){
    if(normChoice == 2){
        funptr = std::make_shared<Funcptr>(&F2_distance);
    }else if(normChoice == 1){
        funptr = std::make_shared<Funcptr>(&F1_distance);
    }else{
        funptr = std::make_shared<Funcptr>(&Fn_distance);
    }

    std::vector<int> contourBound_cur(infer_pts.size(),1);
    unsigned int infer_size = infer_pts.size();

    for(unsigned int i = 0; i < infer_size; i++){
        float min_dist = 1e8f;
        for(unsigned int kk = 0 ; kk < m ; kk++){
            float dist_cur_prevSp_new = (*funptr)(infer_pts[i],ordered_pt[kk]);

            if(dist_cur_prevSp_new < min_dist)
                min_dist = dist_cur_prevSp_new;
            if(min_dist > ordered_radi[kk+1]){
                contourBound_cur[i] = 0;
                break;
            }
        }
    }

    return contourBound_cur;
}


std::vector<int> Ifps::inverse_shell_gpu(
    const std::vector<Eigen::Vector3f> &infer_pts,
    const std::vector<Eigen::Vector3f> &ordered_pt,
    const std::vector<float> &ordered_radi,
    const unsigned int normChoice){

    // std::cout<<"[INFO] SHELL TIME "<<std::endl;
    std::shared_ptr<Timer> timer = std::make_shared<Timer>();

    unsigned int threads = 256;   
    unsigned int blocks = (infer_pts.size() + threads-1) / threads;
    size_t s_size = infer_pts.size();
    size_t m = ordered_pt.size() -1;

    std::vector<int>  contourBound_cur_cpu(s_size,1);
    GpuArray<Eigen::Vector3f> infer_pts_gpu(infer_pts,s_size);
    GpuArray<Eigen::Vector3f> ordered_pt_gpu(ordered_pt,ordered_pt.size());
    GpuArray<float> ordered_radi_gpu(ordered_radi,ordered_radi.size());
    GpuArray<int> contourBound_cur_gpu(contourBound_cur_cpu,s_size);
    GpuArray<float> dist_cur_prevSp_gpu(m);

    timer->start();

    inverse_shell_cu<<<blocks,threads>>>(infer_pts_gpu.ptr(),ordered_pt_gpu.ptr(),ordered_radi_gpu.ptr(),contourBound_cur_gpu.ptr(),s_size,m,normChoice);

    timer->stop();
    // timer->printTime();

    contourBound_cur_gpu.copyToCPU(contourBound_cur_cpu.data(),contourBound_cur_cpu.size());
    return contourBound_cur_cpu;
}


torch::Tensor Ifps::inverse_shell_gpu(
    const torch::Tensor &infer_pts,
    const torch::Tensor &ordered_pt,
    const torch::Tensor &ordered_radi,
    const unsigned int normChoice
){
    // std::cout<<"[INFO] SHELL TORCH TIME "<<std::endl;
    std::shared_ptr<Timer> timer = std::make_shared<Timer>();
    unsigned int threads = 256;   
    unsigned int blocks = (infer_pts.size(0) + threads-1) / threads;

    int s_size = infer_pts.size(0);
    int m = ordered_pt.size(0) -1;

    torch::Tensor contourBound_cur = torch::ones({s_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    timer->start();
    inverse_shell_cu<<<blocks,threads,0,at::cuda::getCurrentCUDAStream()>>>(
        infer_pts.data_ptr<float>(),
        ordered_pt.data_ptr<float>(),
        ordered_radi.data_ptr<float>(),  
        contourBound_cur.data_ptr<int>(),
        s_size,m,normChoice
    );

    timer->stop();
    // timer->printTime();

    return contourBound_cur;
}





