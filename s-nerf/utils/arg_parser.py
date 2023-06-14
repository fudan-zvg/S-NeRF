from argparse import Action
from email.policy import default
from os import access


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    # Rays Settings
    parser.add_argument("--random_sample", action='store_true', 
                        help='ramdom sample rays')  
    parser.add_argument("--N_rgb", type=int, default=512, 
                        help='random rgb ray size')     
    parser.add_argument("--patch_sz", type=int, default=8, 
                        help='rgb patch size (NxN)')
    parser.add_argument("--N_rgb_patch", type=int, default=2, 
                        help='rgb patch num N')
    parser.add_argument("--N_depth", type=int, default=32*32*4, 
                        help='batch size of depth rays (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=128, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_test_ray", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_train", action='store_true', 
                        help='render the train set instead of render_poses path')  
    parser.add_argument("--render_mypath", action='store_true', 
                        help='render the test path')         
    # parser.add_argument("--render_factor", type=int, default=0, 
    #                     help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=1, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    
    # debug
    parser.add_argument("--debug",  action='store_true')

    # new experiment by kangle
    parser.add_argument("--N_iters", type=int, default=200000, 
                        help='number of iters')
    parser.add_argument("--alpha_model_path", type=str, default=None,
                        help='predefined alpha model')
    parser.add_argument("--no_coarse", action='store_true',
                        help="Remove coarse network.")
    parser.add_argument("--train_scene", nargs='+', type=int,
                        help='id of scenes used to train')
    parser.add_argument("--test_scene", nargs='+', type=int,
                        help='id of scenes used to test')
    parser.add_argument("--colmap_depth", action='store_true',
                        help="Use depth supervision by colmap.")
    parser.add_argument("--depth_loss", action='store_true',
                        help="Use depth supervision by colmap - depth loss.")
    parser.add_argument("--depth_lambda", type=float, default=0.1,
                        help="Depth lambda used for loss.")

    parser.add_argument("--weighted_loss", action='store_true',
                        help="Use weighted loss by reprojection error.")
    parser.add_argument("--relative_loss", action='store_true',
                        help="Use relative loss.")

    # Datasettings
    parser.add_argument("--car_sample_n", type=int,
                        help="The Select Car ID")
    parser.add_argument("--block_bg", action="store_true",
                        help="Block The Background")
    parser.add_argument("--smooth_lambda", type=float,default=0.1,
                        help="Smoooth lambda used for loss.")
    parser.add_argument("--proposal_lambda", type=float,default=0.05,
                        help="Proposal lambda used for loss.")
    parser.add_argument('--semantic_loss_type', type=str, default='CE')
    parser.add_argument("--semantic_lambda", type=float,default=0.1,
                        help="Semantic lambda used for loss.")
    parser.add_argument("--pose_refine", action="store_true",
                        help="Refine the camera pose during training")
    parser.add_argument("--reg_loss", action="store_true",
                        help="Regularization loss from mip-nerf 360")
    parser.add_argument("--reg_lambda", type=float,
                        help="Regularization lambda used for loss.")
    parser.add_argument("--depth_conf", action="store_true",
                        help="Reprojection error to calc depth confidence")
    parser.add_argument("--colmap", action="store_true",
                        help="use colmap poses")
    parser.add_argument("--datahold", type=int,
                        help="split the train and test")
    parser.add_argument("--weight_decay_mult", type=float,default=0.,
                    help="split the train and test")
    parser.add_argument("--randomized", action="store_true",
                    help="use randomized intervals")
    parser.add_argument("--H",type=int,default=288)
    parser.add_argument("--W",type=int,default=512)
    parser.add_argument("--coarse_loss_mult",type= float,default=0.1)
    parser.add_argument("--no_warp_sample",type=int,default=1,help='warp sampling or not')
    parser.add_argument("--disable_integration",type=int,default=0,help='disable integration')
    parser.add_argument('--ray_shape',type=str,default='cylinder',help = 'decide the ray shape')
    parser.add_argument('--fn',type=int,default=0,help='choose the warp functions')
    parser.add_argument('--no_align',type=int,default=1)
    parser.add_argument('--max_degree',type=int,default=16)
    parser.add_argument('--radius',type=float,default=3.,help='radius of mipnerf360')
    parser.add_argument('--log',action='store_true',help='use log sampling or not')
    parser.add_argument('--transform_idx',type=int,default=0,help='use log sampling or not')
    parser.add_argument('--real',action='store_true',help='use real distance rendering')
    parser.add_argument('--disparity_depth',action='store_true',help='use disparity depth loss')
    parser.add_argument('--skymask',action = 'store_false',help = 'mask sky')
    parser.add_argument('--bds_factor',type = float, default=0.75,help='bounds factor')
    parser.add_argument('--resume',action = 'store_true',help='resume from ckpt')
    parser.add_argument('--cam_num',type = int ,default=1,help='the number of the cameras included in the images')
    parser.add_argument('--conf_num',type = int,default = 1,help='the number of images to compute the confidence')
    parser.add_argument('--translation',action='store_true',help='use translation for pose refine')
    parser.add_argument('--conf_max',action='store_true',help='use the max confidence')
    parser.add_argument('--near_far',action='store_true',help='use the near and far bounds from depth or not')
    parser.add_argument('--distributed',action='store_true')
    parser.add_argument('--vgg_loss',action='store_true')
    parser.add_argument('--smooth_loss',action='store_true')
    parser.add_argument('--N_patch',type=int,default=8)
    parser.add_argument('--patch_size',type=int,default=8)
    parser.add_argument('--hidden_layer',type=int,default=256,help='the hidden layers dimension')
    parser.add_argument('--rgb_layer',type=int,default=1,help='the num of rgb_layers')
    parser.add_argument('--density_noise',type=float,default=1.)
    parser.add_argument('--backcam',action='store_true')
    parser.add_argument('--flow',action='store_true')
    parser.add_argument('--load_poses',type=str,default=None)
    parser.add_argument('--precompute_conf',action='store_true',help='precompute conf_map')
    parser.add_argument('--far_bound',type=int,default=0)
    parser.add_argument('--half_train',action='store_true')
    parser.add_argument('--seg_mask',action='store_true',help='use mask')
    parser.add_argument('--fulltrain',action='store_true')
    parser.add_argument('--encode_appearance',action='store_true')
    parser.add_argument('--no_reproj',action='store_true')
    parser.add_argument('--no_geometry',action='store_true')
    parser.add_argument('--tau',type=float,default=0.2)
    parser.add_argument('--waymo',action='store_true')
    parser.add_argument('--coarse_depth_mult',type=float,default=0.2)
    parser.add_argument('--proposal_loss',action='store_true')
    parser.add_argument('--N_fine',type=int,default=128)
    parser.add_argument('--semantic',action='store_true')
    parser.add_argument('--semantic_class_num',type=int,default=29)

    ## test
    parser.add_argument("--eval_test",type=int,default=1)
    parser.add_argument("--eval_train",type=int,default=0)
    parser.add_argument("--ckpt",type=int,default=50000)
    parser.add_argument("--test_refine_iter",type=int,default = 0,help='test finetune iteration')
    parser.add_argument('--render_factor',type=int,default=1)
    parser.add_argument('--render_sky',action='store_true')
    parser.add_argument('--half_test',action='store_true')
    return parser
