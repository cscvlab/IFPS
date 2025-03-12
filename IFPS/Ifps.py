import os
import math
import torch
import PyIfps
import numpy as np
import fpsample


class Ifps():
    def __init__(self,args,normChoice,fps_sample_num):
        self.ifps = PyIfps.Ifps()
        self.args = args
        self.normChoice = normChoice
        self.fps_sample_num = fps_sample_num

    def fps(self,dense_pts,filename):
        fpsPath = self.args.dataPath + filename + "/fps_pts_{}_L{}.xyz".format(self.fps_sample_num,self.normChoice)


        if not os.path.exists(fpsPath): 
            dense_pts = np.concatenate((np.zeros((1, 3)), dense_pts), axis=0)
            index = fpsample.fps_sampling(dense_pts, self.fps_sample_num+1,start_idx=0)
  
            self.ordered_pt = dense_pts[index]
            np.savetxt(fpsPath,self.ordered_pt)
            print("[INFO] --Generate {} FPS Points From Point Cloud".format(self.fps_sample_num))
        else:
            self.ordered_pt = np.loadtxt(fpsPath)
            print("[INFO] --Read {} FPS Points ".format(self.args.fps_sample_num))

        return self.ordered_pt


    def inverse_fps(self,filename):
        radiPath = self.args.dataPath + filename + "/radi_{}_L{}.xyz".format(self.fps_sample_num,self.normChoice)
        
        if not os.path.exists(radiPath): 
            print("[INFO] --Generate {} FPS Radius".format(self.fps_sample_num))
            self.ordered_radi =  self.ifps.inverse_fps(self.ordered_pt,self.fps_sample_num,self.normChoice)
            self.ordered_radi = np.array(self.ordered_radi).reshape(-1, 1)

            self.ordered_radi = np.sqrt(self.ordered_radi) + 0.001
            self.ordered_radi = self.ordered_radi * self.ordered_radi
            np.savetxt(radiPath,self.ordered_radi)

        else:
            self.ordered_radi = np.loadtxt(radiPath)
            print("[INFO] --Read {} Fps Radius".format(self.args.fps_sample_num))

        return self.ordered_radi
 

    def inverse_fps_shell(self,infer_pts,type = "GPU"):
        if type == "GPU":
            self.ordered_pt = torch.tensor(self.ordered_pt,dtype = torch.float32).cuda()
            self.ordered_radi = torch.tensor(self.ordered_radi,dtype = torch.float32).cuda()
            infer_pts = torch.tensor(infer_pts,dtype = torch.float32).cuda()

            self.contourBound_cur = self.ifps.inverse_shell_gpu(infer_pts, self.ordered_pt, self.ordered_radi,self.normChoice)

        elif type == "CPU":
            infer_pts = infer_pts.reshape(-1,1)
            self.ordered_pt = self.ordered_pt.reshape(-1,1)

            self.contourBound_cur = self.ifps.inverse_shell_cpu(infer_pts, self.ordered_pt, self.ordered_radi,self.args.fps_sample_num, self.normChoice)

            infer_pts = infer_pts.reshape(-1,3)
            self.contourBound_cur = np.array(self.contourBound_cur)
            

        inner_pts = infer_pts[self.contourBound_cur == 1]
        return inner_pts
    

    def inverse_fps_shell_8way(self,infer_pts,IFPS_SphereTree_path,ChildFirstIndex_path,ChildNum_path,type = "GPU"):
        IFPS_SphereTree = np.loadtxt(IFPS_SphereTree_path)
        ChildFirstIndex = np.loadtxt(ChildFirstIndex_path)
        ChildNum = np.loadtxt(ChildNum_path)

        n = math.ceil(math.log(self.args.fps_sample_num,8)) + 1
        r = np.zeros((n))

        if torch.is_tensor(self.ordered_radi):
            ordered_radi = torch.sqrt(self.ordered_radi)
        else:
            ordered_radi = np.sqrt(self.ordered_radi)

        for i in range(n-1):
            r[i] =  ordered_radi[pow(8,i)]
        r[n-1] = ordered_radi[self.args.fps_sample_num]

        if type == "GPU":
            IFPS_SphereTree = torch.tensor(IFPS_SphereTree,dtype = torch.float32).cuda()
            ChildFirstIndex = torch.tensor(ChildFirstIndex, dtype = torch.int32).cuda()
            ChildNum = torch.tensor(ChildNum, dtype = torch.int32).cuda()
            r = torch.tensor(r,dtype = torch.float32).cuda()
            infer_pts = torch.tensor(infer_pts,dtype = torch.float32).cuda()
 
            contourBound_cur,time = PyIfps.check_treepts_gpu_time(infer_pts,IFPS_SphereTree,ChildFirstIndex,ChildNum,r,n)

        elif type == "CPU":

            infer_pts = infer_pts.reshape(-1,1)
            IFPS_SphereTree = IFPS_SphereTree.reshape(-1,1)
            ChildFirstIndex = ChildFirstIndex.reshape(-1,1)
            ChildNum = ChildNum.reshape(-1,1)

            contourBound_cur,time= PyIfps.check_treepts_cpu_time(infer_pts,IFPS_SphereTree,ChildFirstIndex,ChildNum,r,n)


            infer_pts = infer_pts.reshape(-1,3)
            IFPS_SphereTree = IFPS_SphereTree.reshape(-1,3)
            contourBound_cur = np.array(contourBound_cur)
            
        inner_pts = infer_pts[contourBound_cur == 1]
        return inner_pts
    def inverse_fps_shell_8way_id(self,infer_pts,IFPS_SphereTree_path,IFPS_SphereTree_idx_path,ChildFirstIndex_path,ChildNum_path,type = "GPU"):
        IFPS_SphereTree = np.loadtxt(IFPS_SphereTree_path)
        IFPS_SphereTree_id = np.loadtxt(IFPS_SphereTree_idx_path)
        ChildFirstIndex = np.loadtxt(ChildFirstIndex_path)
        ChildNum = np.loadtxt(ChildNum_path)

        n = math.ceil(math.log(self.args.fps_sample_num,8)) + 1
        r = np.zeros((n))

        if torch.is_tensor(self.ordered_radi):
            ordered_radi = torch.sqrt(self.ordered_radi)
        else:
            ordered_radi = np.sqrt(self.ordered_radi)

        for i in range(n-1):
            r[i] =  ordered_radi[pow(8,i)]
        r[n-1] = ordered_radi[self.args.fps_sample_num]

        if type == "GPU":
            IFPS_SphereTree = torch.tensor(IFPS_SphereTree,dtype = torch.float32).cuda()
            IFPS_SphereTree_id = torch.tensor(IFPS_SphereTree_id,dtype = torch.int32).cuda()
            ChildFirstIndex = torch.tensor(ChildFirstIndex, dtype = torch.int32).cuda()
            ChildNum = torch.tensor(ChildNum, dtype = torch.int32).cuda()
            r = torch.tensor(r,dtype = torch.float32).cuda()
            infer_pts = torch.tensor(infer_pts,dtype = torch.float32).cuda()
 
            contourBound_cur,id,time = PyIfps.check_treepts_gpu_id(infer_pts,IFPS_SphereTree,IFPS_SphereTree_id,ChildFirstIndex,ChildNum,r,n)

            
        inner_pts = infer_pts[contourBound_cur == 1]
        return inner_pts,id.contiguous().reshape(-1,n)
 