import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from Model import VAE_EAD

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument('--data_file', type=str, help='path of input scRNA-seq file.')
parser.add_argument('--transpose', action='store_true', default=False, help='Transpose the input data.')
parser.add_argument('--alpha1', type=float, default=10, help='coefficient for L-1 norm of A.')
parser.add_argument('--beta', type=float, default=1, help='coefficient for KL term.')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
parser.add_argument('--lr_step_size', type=int, default=1, help='step size of LR decay.')
parser.add_argument('--gamma', type=float, default=0.95, help='LR decay factor.')
parser.add_argument("--n_hidden", type=int, default=128)
parser.add_argument("--rep", type=int, default=0)
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--K1", type=int, default=1)
parser.add_argument("--K2", type=int, default=2)
parser.add_argument("--init", type=int, default=1)
parser.add_argument("--nonLinear", type=str, default='tanh')
parser.add_argument("--save_name", type=str, default='/tmp')
class deepsem_embed:
    def __init__(self,opt):
        self.opt = opt
        self.nonLinear = nn.Tanh()
    def initalize_A(self,data):
        num_genes = data.shape[1]
        if self.opt.init == 0:
            A = np.zeros([num_genes, num_genes])
        elif self.opt.init == 1:
            A = np.ones([num_genes, num_genes]) / (num_genes - 1) + (np.random.rand(num_genes * num_genes) * 0.0002).reshape(
                [num_genes, num_genes])
        for i in range(len(A)):
            A[i, i] = 0
        return A

    def init_data(self):
        data = pd.read_csv(self.opt.data_file, header=0, index_col=0).transpose()
        if self.opt.transpose:
            data = data.T
        gene_name = data.columns
        data_values = data.values
        Dropout_Mask = (data_values != 0).astype(float)
        means = []
        stds = []
        for i in range(data_values.shape[1]):
            tmp = data_values[:, i]
            means.append(tmp[tmp != 0].mean())
            stds.append(tmp[tmp != 0].std())
        means = np.array(means)
        stds = np.array(stds)
        stds[np.isnan(stds)] = 0
        stds[np.isinf(stds)] = 0
        data_values = (data_values - means) / (stds)
        data_values[np.isnan(data_values)] = 0
        data_values[np.isinf(data_values)] = 0
        data_values = np.maximum(data_values,-10)
        data_values = np.minimum(data_values,10)
        data = pd.DataFrame(data_values, index=data.index, columns=data.columns)
        num_genes, num_nodes = data.shape[1], data.shape[0]
        TF_mask = np.zeros([num_genes, num_genes])
        for i, item in enumerate(data.columns):
            for j, item2 in enumerate(data.columns):
                if i == j:
                    continue
                TF_mask[i, j] = 1
        feat_train = torch.FloatTensor(data.values)
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask))
        dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=1)
        return dataloader,  num_nodes, num_genes, data, TF_mask, gene_name, Dropout_Mask


    def train_model(self,):
        opt =self.opt
        dataloader, num_nodes, num_genes, data, TFmask2, gene_name, Dropout_Mask = self.init_data()
        adj_A_init  = self.initalize_A(data)
        vae = VAE_EAD(adj_A_init, 1, opt.n_hidden, opt.K,self.nonLinear).float().cuda()
        Tensor = torch.cuda.FloatTensor
        optimizer = optim.RMSprop(vae.parameters(), lr=opt.lr)
        optimizer2 = optim.RMSprop([vae.adj_A], lr=opt.lr * 0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.gamma)
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
                loss, loss_rec, loss_gauss, loss_cat, dec, y, hidden = vae(inputs,dropout_mask=dropout_mask.cuda(),temperature=temperature,opt=opt)
                sparse_loss = opt.alpha * torch.mean(torch.abs(vae.adj_A))
                loss = loss + sparse_loss
                loss = loss
                loss.backward()
                mse_rec.append(loss_rec.item())
                loss_all.append(loss.item())
                loss_kl.append(loss_gauss.item() + loss_cat.item())
                loss_sparse.append(sparse_loss.item())
                if epoch % (opt.K1+opt.K2) < opt.K1:
                    optimizer.step()
                else:
                    optimizer2.step()
            print('epoch:', epoch, 'loss',
                  np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_kl), 'sparse_loss:',
                  np.mean(loss_sparse))
            scheduler.step()
        data_ids = []
        embeds1 = []
        n_rep = 1
        for _ in range((n_rep)):
            for i, data_batch in enumerate(dataloader, 0):
                optimizer.zero_grad()
                inputs, data_id, dropout_mask = data_batch
                inputs = Variable(inputs.type(Tensor))
                loss, loss_rec, loss_gauss, loss_cat, dec, y, hidden = vae(inputs,dropout_mask=dropout_mask.cuda(),temperature=temperature,opt=opt)
                data_ids.append(data_id.detach().numpy())
                embeds1.append(hidden.cpu().detach().numpy())
        data_ids = np.hstack(data_ids)
        embeds1 = np.vstack(embeds1)
        import pickle as pkl
        pkl.dump([data_ids,embeds1],open(str(epoch)+'embed_out.pkl'+str(time.time()),'wb'))


