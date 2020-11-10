'''

Utils functions for the main programme
Gaussian Mixture Variational Autoencoder

'''

import argparse
from bunch import Bunch
import os


def check_args(args):
    '''
	This method check that the values that are provided are correct
	'''
    try:
        assert args.epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --z_dim
    try:
        assert args.z_dim >= 1
    except:
        print('dimension of noise vector must be larger than or equal to one')

    try:
        assert args.restore == 0 or args.restore == 1
    except:
        print('restore flag must be 0 or 1')

    try:
        assert args.cuda == 0 or args.cuda == 1
    except:
        print('cuda flag must be 0 or 1')

    try:
        assert args.remove == 0 or args.remove == 1
    except:
        print('remove flag must be 0 or 1')

    try:
        assert args.verbose == 0 or args.verbose == 1
    except:
        print('verbose flag must be 0 or 1')

    return args


def get_args():
    ## Input Parameters
    parser = argparse.ArgumentParser(description='PyTorch Implementation of CGMVAE')

    # Dataset
    parser.add_argument('--name_extension', type=str, default='2020-01-26 10:59:52',
                        help='Pattern to be found to obtain the data (default: 2020-01-26 10:59:52)')
    parser.add_argument('--dataset_name', type=str, default='dataset13_2020-01-26_10:59:52',
                        help='Dataset used (from the seq2seq) for obtaining the data (default: dataset12)')
    parser.add_argument('--dataroot', type=str, default='./data/4_decoder_context/',
                        help='Root to load dataset (default: ./data/3_decoder_output_enc/)')
    parser.add_argument('--train_dataroot', type=str,
                        default='../TransformersNLP/results/dataset13/bert_base_retrained_dataset13/lastHiddenState_30000_train.hdf5',
                        help='Root to load dataset')
    parser.add_argument('--test_dataroot', type=str,
                        default='../TransformersNLP/results/dataset13/bert_base_retrained_dataset13/lastHiddenState_1000_test.hdf5',
                        help='Root to load dataset ')

    # GPU
    parser.add_argument('--cuda', type=int, default=0, help='use of cuda (default: 1)')
    parser.add_argument('--device', type=int, default=0, help='set gpu device to use (default: 0)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for training each bunch of data (default: 200)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of the mini-batch used on each iteration (default: 64)')
    parser.add_argument('--l_rate', type=float, default=1e-5,
                        help='Learning rate of the optimization function (default: 0.000001)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate in the training (default: 0.3)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay in the training optimizer (default: 00)')

    # Architecture
    parser.add_argument('--hidden_dim', type=int, default=1500,
                        help='Number of neurons of each dense layer (default: 1500)')
    parser.add_argument('--sigma', type=float, default=1e-2,
                        help='Parameter that defines the variance of the output Gaussian distribution (default: 0.0001)')
    parser.add_argument('--z_dim', type=int, default=150, help='Dimension of the latent variable z (default: 150)')
    parser.add_argument('--w_dim', type=int, default=50, help='Dimension of the latent variable w (default: 50)')
    parser.add_argument('--K', type=int, default=20, help='Number of modes of the latent variable z (default: 20)')
    parser.add_argument('--layers', type=int, default=6, help='Number of layers in the networks (default: 3)')

    # Results
    parser.add_argument('--remove', type=int, default=1, help='Remove old checkpoint files (default: 0)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory name to save the checkpoints (default: checkpoint)')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated sentences (default: results)')
    parser.add_argument('--board_dir', type=str, default='summary',
                        help='Directory name to save in tensorboard (default: summary)')

    # Others
    parser.add_argument('--train', type=int, default=0, help='Flag to set train (default: 1)')
    parser.add_argument('--summary', type=int, default=1, help='Flag to set TensorBoard summary (default: 1)')
    parser.add_argument('--plot', type=int, default=0, help='Flag to plot training curves (default: 0)')
    parser.add_argument('--restore', type=int, default=0, help='Flag to restore model (default: 0)')
    parser.add_argument('--results', type=int, default=0, help='Flag to get results (default: 0)')
    parser.add_argument('--verbose', type=int, default=1, help='print extra information at every epoch (default: 1)')
    parser.add_argument('--extra', type=str, default='', help='Extra name to identify the model (default: '')')
    parser.add_argument('--step_restore', type=int, default=200, help='Global step to be loaded (default: 200)')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='Every step the model is saved (default: 10)')
    parser.add_argument('--option', type=int, default=1, help='seq2seq option (default: 1)')

    args = parser.parse_args()

    return check_args(args)


def get_model_name(config):
    model_name = 'GMVAE_' + \
                 str(config.option) + '_' + \
                 str(config.sigma).replace('.', '') + '_' + \
                 str(config.z_dim) + '_' + \
                 str(config.w_dim) + '_' + \
                 str(config.K) + '_' + \
                 str(config.hidden_dim) + '_' + \
                 str(config.layers) + '_' + \
                 str(config.dropout).replace('.', '') + '_' + \
                 str(config.l_rate).replace('.', '')
    return model_name


def get_config_and_flags(args):
    config = Bunch(args)

    config.model_name = get_model_name(config)

    if (config.extra is not ''):
        config.model_name += '_' + config.extra

    config.board_dir = os.path.join("experiments/" + config.dataset_name + '/' + config.board_dir + "/",
                                    config.model_name)
    config.checkpoint_dir = os.path.join("experiments/" + config.dataset_name + '/' + config.checkpoint_dir + "/",
                                         config.model_name)
    config.result_dir = os.path.join("experiments/" + config.dataset_name + '/' + config.result_dir + "/",
                                     config.model_name)

    flags = Bunch()
    flags.train = args['train']
    flags.summary = args['summary']
    flags.restore = args['restore']
    flags.verbose = args['verbose']
    flags.results = args['results']

    return config, flags


def create_dirs(dirs):
    '''
	dirs - a list of directories to create if these directories are not found
	:param dirs:
	:return exit_code: 0:success -1:failed
	'''
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def save_args(args, board_dir):
    my_file = board_dir + '/' + 'my_args.txt'
    args_string = str(args).replace(', ', ' --')
    with open(my_file, 'a+') as file_:
        file_.write(args_string)


def plot_results(images, N, text_title=''):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.suptitle(text_title, fontsize=16)
    for i in range(N):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i, :, :], cmap='gray')
    plt.show()
