import argparse

def parse_args():
    parse =  argparse.ArgumentParser()
    parse.add_argument('--batch_size',type = int,default= 8192,help = 'input batch size')
    parse.add_argument('--epoch',type = int,default= 100,help = 'number of epoch in training')
    parse.add_argument('--file_path',type = str,default= None ,help = 'mesh(obj) path of meshPath')
    parse.add_argument('--file_name',type = str,default= None ,help = 'mesh(obj) name of meshPath')
    parse.add_argument('--dir',type = int,default= None ,help = 'the direction of rays')
    parse.add_argument('--gpu',type = str,default='0',help = 'specify gpu device')
    parse.add_argument('--pretrain',type = str,default=None,help = 'whether use pertrain model')
    parse.add_argument('--mask_pretrain',type = str,default=None,help = 'whether use mask pertrain model')
    parse.add_argument('--train_metric',type = str,default=True,help = 'whether evaluate on traning dataset')
    parse.add_argument('--model_name',type = str,default= 'HeightModel',help = 'model name')
    parse.add_argument('--learning_rate',type = float,default= 0.001,help = 'learning rate in training')
    parse.add_argument('--optimizer',type = str,default='adam',help = 'optimizer for training')
    parse.add_argument('--decay_rate',type = float,default=1e-4,help = 'decay raye of learning rate')
    parse.add_argument('--use_embedder',type = bool,default= False,help = 'whether use embedder of point')
    parse.add_argument('--multries_xyz',default= 10,help = 'log2 of ,ax freq for positional encoding(3D location)')

    '''sample setting'''
    # e6
    parse.add_argument('--mesh_num_sample',type = int,default= 1000000,help = 'sample number of mesh')
    parse.add_argument('--fps_sample_num',type = int,default=256,help = 'sample fps of points')
    parse.add_argument('--axis_upsample_times',type = int,default= 8 ,help = 'upsample_times of pts in the plane')
    # parse.add_argument('--full_upsample_times',type = int,default= 16 ,help = 'upsample_times of pts in the plane')


    '''path setting'''
    parse.add_argument('--meshPath', help='path to input mesh/ folder of meshes', type=str,
                        default= None)
    parse.add_argument('--dataPath', help='path to data folder of gt', type=str,
                        default= None)
    parse.add_argument('--ExpPath', help='path to exp folder of train', type=str,
                        default= None)


    parse.add_argument('--testPath', help='path to test folder of pred', type=str,
                        default= None)
    parse.add_argument('--dictPath', help='path to test folder of pred', type=str,
                        default= None)

    parse.add_argument('--experiment_path', help='path to exp folder of train', type=str,
                        default=None)

    return parse.parse_args()