import os
import torch
import datetime
import numpy as np
from Ifps import Ifps
from utils.options import parse_args
from utils.utils import init_path,sample_mesh_surface

if __name__ == "__main__":
    args = parse_args()
    ifps = Ifps(args,normChoice = 2,fps_sample_num = args.fps_sample_num)

    args.meshPath = "./dataset/thingi32_normalization/"

    init_path(args)

    inferptsPath = "./inferPts.xyz"
    if not os.path.exists(inferptsPath):
        infer_pts = np.random.rand(1000000,3) * 2 - 1
        np.savetxt(inferptsPath,infer_pts)
        print("[INFO] Save Infer Pts")
    else:
        infer_pts = np.loadtxt(inferptsPath)
        print("[INFO] Read Infer Pts")

    
    file_path = '441708.obj'
    file_name = '441708'

    print("[INFO] {}:".format(file_name))

    # sample mesh points
    print("[INFO]------------SAMPLE MESH SURFACE--------------------")
    dense_pts, dense_normals = sample_mesh_surface(args,file_path,mesh_num_sample = args.mesh_num_sample)

    # fps
    print("[INFO]--------------------FPS----------------------------")
    ordered_pt= ifps.fps(dense_pts,filename = file_name)
    ordered_radi = ifps.inverse_fps(filename = file_name)

    # vanilla shell
    print("[INFO]-------------Vanilla SHELL----------------------------")
    res = 256
    x = np.linspace(-1,1,res)
    infer_pts = np.array([(i,j,k) for i in x for j in x for k in x] )

    # inner_pts = ifps.inverse_fps_shell(infer_pts,type = "GPU")
    # print("[INFO] (Vanilla) Inner Shell Points Number: ",inner_pts.shape[0])
    # np.savetxt("inner_pts.xyz",inner_pts.detach().cpu().numpy())

    # 8-way shell
    print("[INFO]-------------8way SHELL----------------------------")
    infer_pts = ifps.ordered_pt
    IFPS_SphereTree_path = args.dataPath + file_name + "/tree/" + "IFPS_{}_{}_balanced.txt".format(file_name,args.fps_sample_num)
    ChildFirstIndex = args.dataPath + file_name + "/tree/" + "IFPS_{}_{}_childindex_balanced.txt".format(file_name,args.fps_sample_num)
    ChildNum = args.dataPath + file_name + "/tree/" + "IFPS_{}_{}_childnum_balanced.txt".format(file_name,args.fps_sample_num)

    # inner_pts = ifps.inverse_fps_shell_8way(infer_pts,IFPS_SphereTree_path = IFPS_SphereTree_path,ChildFirstIndex_path = ChildFirstIndex,
    #                                         ChildNum_path = ChildNum,type = "GPU")
    # print("[INFO] (8Way) Inner Shell Points Number: ",inner_pts.shape[0])


    IFPS_SphereTree_idx_path = args.dataPath + file_name + "/tree/" + "IFPS_{}_{}_tree_inum.txt".format(file_name,args.fps_sample_num)
    inner_pts,id = ifps.inverse_fps_shell_8way_id(infer_pts,IFPS_SphereTree_path = IFPS_SphereTree_path,
                                              IFPS_SphereTree_idx_path = IFPS_SphereTree_idx_path,
                                               ChildFirstIndex_path = ChildFirstIndex,
                                            ChildNum_path = ChildNum,type = "GPU")
    id = id.detach().cpu().numpy()
    print(id[:10])
    # np.savetxt("8way_inner_pts.xyz",inner_pts.detach().cpu().numpy())
    # print("[INFO] (8Way) Inner Shell Points Number: ",inner_pts.shape[0])



