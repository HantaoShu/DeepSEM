import argparse

from src.DeepSEM_cell_type_non_specific_GRN_model import non_celltype_GRN_model
from src.DeepSEM_cell_type_specific_GRN_model import celltype_GRN_model
from src.DeepSEM_generation_model import deepsem_generation
from src.DeepSEM_embed_model import deepsem_embed

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--task", type=str, default='celltype_GRN', help="task")
parser.add_argument("--setting", type=str, default='default', help="task")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument('--data_file', type=str, help='path of input scRNA-seq file.')
parser.add_argument('--net_file', type=str, help='path of input truth net file.')
parser.add_argument('--transpose', action='store_true', default=False, help='Transpose the input data.')
parser.add_argument('--alpha', type=float, default=90, help='coefficient for L-1 norm of A.')
parser.add_argument('--beta', type=float, default=1, help='coefficient for KL term.')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
parser.add_argument('--lr_step_size', type=int, default=1, help='step size of LR decay.')
parser.add_argument('--gamma', type=float, default=0.99, help='LR decay factor.')
parser.add_argument("--n_hidden", type=int, default=128)
parser.add_argument("--rep", type=int, default=0)
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--K1", type=int, default=1)
parser.add_argument("--K2", type=int, default=2)
parser.add_argument("--nonLinear", type=str, default='tanh')
parser.add_argument("--save_name", type=str, default='/tmp')
opt = parser.parse_args()
if opt.task=='non_celltype_GRN':
    if opt.setting=='default':
        opt.beta = 1
        opt.alpha= 90
        opt.K1 = 1
        opt.K2 = 2
        opt.n_hidden =128
        opt.gamma = 0.99
        opt.lr = 1e-4
        opt.lr_step_size=0.99
        opt.batch_size=64
    model = non_celltype_GRN_model(opt)
    model.train_model()
elif opt.task=='celltype_GRN':
    if opt.setting=='default':
        opt.beta = 0.1
        opt.alpha= 10
        opt.K1 = 1
        opt.K2 = 2
        opt.n_hidden =128
        opt.gamma = 0.99
        opt.lr = 1e-4
        opt.lr_step_size=0.99
        opt.batch_size=64
    model = celltype_GRN_model(opt)
    model.train_model()
elif opt.task=='generate':
    if opt.setting=='default':
        opt.beta = 2.5
        opt.alpha= 10
        opt.K1 = 1
        opt.K2 = 2
        opt.n_hidden =128
        opt.gamma = 0.99
        opt.lr = 1e-4
        opt.lr_step_size=0.99
        opt.batch_size=64
    model = deepsem_generation(opt)
    model.train_model()
elif opt.task == 'embedding':
    if opt.setting == 'default':
        opt.beta = 1
        opt.alpha = 10
        opt.K1 = 1
        opt.K2 = 2
        opt.n_hidden = 128
        opt.gamma = 0.99
        opt.lr = 1e-4
        opt.lr_step_size = 0.99
        opt.batch_size = 64
    model = deepsem_embed(opt)
    model.train_model()