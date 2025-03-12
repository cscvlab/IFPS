import os
import torch
import trimesh
import h5py
import logging
import datetime
import numpy as np
from utils.sample_surface import sample_surface



def init_dir(args):
    file_name = args.file_name
    dir = args.dir

    experiment_path = create_directory(os.path.join(args.ExpPath, file_name))
    experiment_dir = create_directory(os.path.join(experiment_path,file_name+ "_"+ str(dir) + "/"))
    checkpoints_dir = create_directory(os.path.join(experiment_dir,'checkpoints'))
    log_dir = create_directory(os.path.join(experiment_dir,'logs'))

    return checkpoints_dir,experiment_path,experiment_dir


def init_path(args):
    path = os.path.abspath(os.path.dirname(__file__))
    path = path[:path.rindex('utils')]
    # print(path)
    # path = "/"

    if (args.meshPath == None):
        meshPath = path + "datasets/thingi32_normalization/"
        if (not os.path.exists(meshPath)):
            os.mkdir(meshPath)
        args.meshPath = meshPath

    if (args.ExpPath == None):
        ExpPath = path + "experiments/"
        if (not os.path.exists(ExpPath)):
            os.mkdir(ExpPath)
        args.ExpPath = ExpPath

    if (args.dataPath == None):
        dataPath = path + "data/"
        if (not os.path.exists(dataPath)):
            os.mkdir(dataPath)
        args.dataPath = dataPath


def init_log(args,experiment_path):
    '''log'''
    logger = logging.getLogger(args.model_name)
    # logger = logging.getLogger('UODF')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(experiment_path + 'logs/train_%s_' % str(args.model_name)
                                       + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')) + '.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('-------------------------------------------traning-------------------------------------')
    logger.info('Parameter...')
    logger.info(args)
    '''Data loading'''
    logger.info('Load dataset...')
    return logger


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))

def save_checkpoint(epoch,train_accuracy,test_accuracy,model,optimizer,path,modelnet = 'checkpoint'):
    savepath = path + '/%s-%f-%04d.pth' %(modelnet,test_accuracy,epoch)
    state = {
        'epoch':epoch,
        'train_accuracy':train_accuracy,
        'test_accuracy':test_accuracy,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
    }
    torch.save(state,savepath)


def save_h5(path ,data):
    with h5py.File(path, 'w') as f:
        f["data_old"] = data

def load_h5(filename):
    f = h5py.File(filename, 'r')
    data = np.array(f.get('data_old'))
    return data

def findRow(mat,row):
    return np.where((mat == row).any(1))[0]

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("\033[0;31;40m[Create Directory]\033[0m{}".format(path))
    return path

def save_dict_to_txt(save_dict,save_path):
    with open(save_path, 'w') as f:
        for key, value in save_dict.items():
            f.write(key)
            f.write(': ')
            f.write(str(value))
            f.write('\n')
        f.close()

def sample_mesh_surface(args,filename,mesh_num_sample = 100000):
    f = filename.split(".")[0]
    if not os.path.exists(args.dataPath + f):
        os.makedirs(args.dataPath + f)

    if not os.path.exists(args.dataPath + f + "/tree/"):
        os.makedirs(args.dataPath + f + "/tree/" )

    if not os.path.exists(args.dataPath + f + "/mesh_sample_points_normals.h5"):
        mesh = trimesh.load(args.meshPath + filename)
        V, F = torch.tensor(mesh.vertices), torch.tensor(mesh.faces)
        dense_pts, normals = sample_surface(V, F, num_samples=mesh_num_sample)
        dense_pts, normals = dense_pts.numpy(),normals.numpy()

        points_normals = np.concatenate((dense_pts,normals),axis = 1)
        save_h5(args.dataPath + f + "/mesh_sample_points_normals.h5",points_normals)
        print("[INFO] Sample {} points from Models(Mesh Points)".format(mesh_num_sample))
    else:
        points_normals = load_h5(args.dataPath + f + "/mesh_sample_points_normals.h5")
        dense_pts,normals = points_normals[:,:3],points_normals[:,3:]
        print("[INFO] Read {} points from Models(Mesh Points)".format(mesh_num_sample))
    return dense_pts,normals


class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)