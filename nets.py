from __future__ import print_function
import torch
import numpy as np
# from fcn import FCN
from unet import FCN

# define the clapping threshold for probability normalization.
STAB_NB = 1e-15


def newton_sol(g_mean, g_sigma, w_comp):
    '''
    This function solves the quadratic CRF: 1/2 xT H x + pT x. Assume g_mean has shape: bn,  x_len. w_comp is a torch parameterlist.
    '''
    x_len = g_mean.size(1)
    # The Hessian is divided into two parts: pairwise and unary.
    hess_pair = torch.diag(-2.*w_comp.repeat(x_len-1), diagonal=-1) + torch.diag(-2.*w_comp.repeat(x_len-1),
                    diagonal=1) + torch.diag(torch.cat((2.*w_comp, 4.*w_comp.repeat(x_len-2), 2.*w_comp), dim=0),
                        diagonal=0)
    # hess_pair = torch.stack(hess_pair)
    # pairwise parts are the same across patches within a batch
    if g_mean.is_cuda:
        hess_pair_batch = torch.stack([hess_pair]*g_sigma.size(0)).cuda()
    else:
        hess_pair_batch = torch.stack([hess_pair]*g_sigma.size(0))
    # get reverse of sigma array
    g_sigma_rev = 1./g_sigma
    # convert sigma reverse array to diagonal matrices
    g_sigma_eye = torch.diag_embed(g_sigma_rev)
    # sum up two parts
    hess_batch = hess_pair_batch + g_sigma_eye
    # compute inverse of Hessian
    hess_inv_batch = torch.inverse(hess_batch)
    # generate the linear coefficient P
    p = g_mean/g_sigma
    # solve it globally
    solution = torch.matmul(hess_inv_batch, p.unsqueeze(-1)).squeeze(-1)

    return solution


def normalize_prob(x):
    '''Normalize prob map to [0, 1]. Numerically, add 1e-6 to all. Assume the last dimension is prob map.'''
    x_norm = (x - x.min(-1, keepdim=True)
              [0]) / (x.max(-1, keepdim=True)[0] - x.min(-1, keepdim=True)[0])
    x_norm += 1e-3
    return x_norm


def gaus_fit(x, tr_flag=True):
    '''This module is designed to regress Gaussian function. Weighted version is chosen. The input tensor should
    have the format: BN,  X_LEN, COL_LEN.'''
    bn,  x_len, col_len = tuple(x.size())
    col_ind_set = torch.arange(col_len).expand(
        bn,  x_len, col_len).double()
    if x.is_cuda:
        col_ind_set = col_ind_set.cuda()
    y = x.double()
    lny = torch.log(y).double()
    y2 = torch.pow(y, 2).double()
    x2 = torch.pow(col_ind_set, 2).double()
    sum_y2 = torch.sum(y2, dim=-1)
    sum_xy2 = torch.sum(col_ind_set * y2, dim=-1)
    sum_x2y2 = torch.sum(x2 * y2, dim=-1)
    sum_x3y2 = torch.sum(torch.pow(col_ind_set, 3) * y2, dim=-1)
    sum_x4y2 = torch.sum(torch.pow(col_ind_set, 4) * y2, dim=-1)
    sum_y2lny = torch.sum(y2 * lny, dim=-1)
    sum_xy2lny = torch.sum(col_ind_set * y2 * lny, dim=-1)
    sum_x2y2lny = torch.sum(x2 * y2 * lny, dim=-1)
    b_num = (sum_x2y2**2*sum_xy2lny - sum_y2*sum_x4y2*sum_xy2lny + sum_xy2*sum_x4y2*sum_y2lny +
             sum_y2*sum_x3y2*sum_x2y2lny - sum_x2y2*sum_x3y2*sum_y2lny - sum_xy2*sum_x2y2*sum_x2y2lny)
    c_num = (sum_x2y2lny*sum_xy2**2 - sum_xy2lny*sum_xy2*sum_x2y2 - sum_x3y2*sum_y2lny*sum_xy2 +
             sum_y2lny*sum_x2y2**2 - sum_y2*sum_x2y2lny*sum_x2y2 + sum_y2*sum_x3y2*sum_xy2lny)
    c_num[(c_num < STAB_NB) & (c_num > -STAB_NB)
          ] = torch.sign(c_num[(c_num < STAB_NB) & (c_num > -STAB_NB)]) * STAB_NB
    mu = -b_num / (2.*c_num)

    c_din = sum_x4y2*sum_xy2**2 - 2*sum_xy2*sum_x2y2*sum_x3y2 + \
        sum_x2y2**3 - sum_y2*sum_x4y2*sum_x2y2 + sum_y2*sum_x3y2**2
    sigma_b_sqrt = -0.5*c_din/c_num
    sigma_b_sqrt[sigma_b_sqrt < 1] = 1
    sigma = sigma_b_sqrt
    if not tr_flag:
        mu[mu >= col_len-1] = col_len-1
        mu[mu <= 0] = 0.
    if torch.isnan(mu).any():
        print("mu be NaN")
    if torch.isnan(sigma).any():
        print("sigma be NaN")
    mu = mu.float()
    sigma = sigma.float()

    return mu, sigma


class SurfNet(torch.nn.Module):
    def __init__(self, wt_init=1e-5, depth=5, start_filts=8):
        super(SurfNet, self).__init__()
        self.U_net = FCN(num_classes=1, in_channels=3, depth=depth,
                         start_filts=start_filts, up_mode='transpose')
        self.w_comp = torch.nn.Parameter(torch.ones(1)*wt_init)

    def forward(self, x, tr_flag=True, U_net_only=False):
        x = self.U_net(x).squeeze(1).permute(0, 2, 1)
        if U_net_only:
            x = torch.nn.functional.log_softmax(x, dim=-1)
            return x
        else:
            x = torch.nn.functional.softmax(x, dim=-1)
            x = normalize_prob(x)
            if tr_flag:
                mean, sigma = gaus_fit(x, tr_flag=True)
            else:
                mean, sigma = gaus_fit(x, tr_flag=False)
            output = newton_sol(mean, sigma, self.w_comp)
            if tr_flag:
                return output
            else:
                return output, mean, sigma
