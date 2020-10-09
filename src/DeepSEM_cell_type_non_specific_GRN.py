import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from utils import evaluate, extractEdgesFromMatrix
from Model import VAE_EAD

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=180, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument('--data_file', type=str, help='path of input scRNA-seq file.')
parser.add_argument('--net_file', type=str, help='path of input truth net file.')
parser.add_argument('--transpose', action='store_true', default=False, help='Transpose the input data.')
parser.add_argument('--alpha1', type=float, default=0, help='coefficient for L-1 norm of A.')
parser.add_argument('--beta', type=float, default=0.9, help='coefficient for KL term.')
parser.add_argument('--alpha2', type=float, default=0, help='coefficient for L-1 term of A which is not out from TF.')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
parser.add_argument('--lr_step_size', type=int, default=1, help='step size of LR decay.')
parser.add_argument('--gamma', type=float, default=0.99, help='LR decay factor.')
parser.add_argument("--n_hidden", type=int, default=128)
parser.add_argument("--rep", type=int, default=0)
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--K1", type=int, default=2)
parser.add_argument("--K2", type=int, default=1)
parser.add_argument("--init", type=int, default=1)
parser.add_argument("--nonLinear", type=str, default='tanh')
parser.add_argument("--save_name", type=str, default='/tmp')
opt = parser.parse_args()
np.random.seed(opt.rep)
torch.manual_seed(opt.rep)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.rep)

Tensor = torch.cuda.FloatTensor
if opt.nonLinear =='tanh':
    nonLinear = nn.Tanh()
elif opt.nonLinear == 'lrelu':
    nonLinear = nn.LeakyReLU(negative_slope=0.1)

try:
    os.mkdir(opt.save_name)
except:
    print('dir exist')
def initalize_A(data):
    num_genes = data.shape[1]
    if opt.init == 0:
        A = np.zeros([num_genes, num_genes])
    elif opt.init == 1:
        A = np.ones([num_genes, num_genes]) / (num_genes - 1) + (np.random.rand(num_genes * num_genes) * 0.0002).reshape(
            [num_genes, num_genes])
    for i in range(len(A)):
        A[i, i] = 0
    return A


def init_data():
    Ground_Truth = pd.read_csv(opt.net_file, header=0)
    data = pd.read_csv(opt.data_file, header=0, index_col=0).transpose()
    if opt.transpose:
        data = data.T
    gene_name = data.columns
    data_values = data.values
    Dropout_Mask = (data_values != 0).astype(float)
    data_values = (data_values - data_values.mean(0)) / (data_values.std(0))
    data = pd.DataFrame(data_values, index=data.index, columns=data.columns)
    TF = set(Ground_Truth['Gene1'])
    All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])
    num_genes, num_nodes = data.shape[1], data.shape[0]
    Evaluate_Mask = np.zeros([num_genes, num_genes])
    TF_mask = np.zeros([num_genes, num_genes])
    for i, item in enumerate(data.columns):
        for j, item2 in enumerate(data.columns):
            if i == j:
                continue
            if item2 in TF and item in All_gene:
                Evaluate_Mask[i, j] = 1
            if item2 in TF:
                TF_mask[i, j] = 1
    feat_train = torch.FloatTensor(data.values)
    train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                               torch.FloatTensor(Dropout_Mask))
    dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    truth_df = pd.DataFrame(np.zeros([num_genes, num_genes]), index=data.columns, columns=data.columns)
    for i in range(Ground_Truth.shape[0]):
        truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
    A_truth = truth_df.values
    idx_rec, idx_send = np.where(A_truth)
    truth_edges = set(zip(idx_send, idx_rec))
    return dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TF_mask, gene_name, Dropout_Mask


def train_model(dataloader, truth_edges, opt, Evaluate_Mask):
    adj_A_init  = initalize_A(data)
    vae = VAE_EAD(adj_A_init, 1, opt.n_hidden, opt.K,nonLinear).float().cuda()
    optimizer = optim.RMSprop(vae.parameters(), lr=opt.lr)
    optimizer2 = optim.RMSprop([vae.adj_A], lr=opt.lr * 0.2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.gamma)
    best_Epr = 0
    vae.train()
    print(vae)
    for epoch in range(opt.n_epochs):
        loss_all, mse_rec, loss_kl, data_ids, loss_tfs, loss_sparse = [], [], [], [], [], []
        if epoch % (opt.K1+opt.K2) < opt.K1:
            vae.adj_A.requires_grad = False
        else:
            vae.adj_A.requires_grad = True
        for i, data_batch in enumerate(dataloader, 0):
            optimizer.zero_grad()
            inputs, data_id, dropout_mask = data_batch
            inputs = Variable(inputs.type(Tensor))
            data_ids.append(data_id.cpu().detach().numpy())
            temperature = max(0.95 ** epoch, 0.5)
            loss, loss_rec, loss_gauss, loss_cat, dec, y, hidden = vae(inputs,dropout_mask=None,temperature=temperature,opt=opt)
            sparse_loss = opt.alpha1 * torch.mean(torch.abs(vae.adj_A))
            loss = loss + sparse_loss
            loss_tf = torch.sum(torch.Tensor(1 - TFmask2).cuda() * torch.abs(vae.adj_A) * opt.alpha2) / torch.sum(
                torch.Tensor(1 - TFmask2).cuda())
            loss = loss + loss_tf
            loss.backward()
            mse_rec.append(loss_rec.item())
            loss_tfs.append(loss_tf.item())
            loss_all.append(loss.item())
            loss_kl.append(loss_gauss.item() + loss_cat.item())
            loss_sparse.append(sparse_loss.item())
            if epoch % (opt.K1+opt.K2) < opt.K1:
                optimizer.step()
            else:
                optimizer2.step()
        scheduler.step()
        if epoch % (opt.K1+opt.K2) >= opt.K1:
            Ep, Epr =  evaluate(vae.adj_A.cpu().detach().numpy(), truth_edges, Evaluate_Mask)
            best_Epr = max(Epr, best_Epr)
            print('epoch:', epoch, 'Ep:', Ep, 'Epr:', Epr, 'loss:',
                  np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_kl), 'sparse_loss:',
                  np.mean(loss_sparse))

        if epoch%10 ==0:
            extractEdgesFromMatrix(vae.adj_A.cpu().detach().numpy(), gene_name,TFmask2).to_csv(
            opt.save_name + '/rep_' + str(opt.rep) + '_epoch_' + str(epoch) + '.tsv', sep='\t', index=False)

if __name__ == '__main__':
    dataloader, TFmask, num_nodes, num_genes, data, truth_edges, TFmask2, gene_name, ori_mask = init_data()
    train_model(dataloader, truth_edges, opt, TFmask)
