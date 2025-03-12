#pragma once
#include <iostream>

class Timer{
private:
    cudaEvent_t Timestart,Timestop;
public:
    float time;
    void start(){
        cudaEventCreate(&Timestart);
        cudaEventCreate(&Timestop);
        cudaEventRecord(Timestart,0);
    }

    void stop(){
        cudaEventRecord(Timestop,0);
        cudaEventSynchronize(Timestop);
        cudaEventElapsedTime(&time,Timestart,Timestop);
        cudaEventDestroy(Timestart);
        cudaEventDestroy(Timestop);
    }

    void printTime(){
        std::cout<<"Kernel RunTime : "<< time <<" ms" <<std::endl;
    }


};