#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:04:11 2023

@author: yz23558
"""

#Oct. 3 
#@YZhang
import torch as tc
import quimb.tensor as qtn
import quimb as qu
from hamiltonian import model_mpo
from mpi4py import MPI
import torch
show_progress_bar=False
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  
from functions import*
from torch.autograd import Variable
import pickle 
def norm_fn(psi):
    # parametrize our tensors as isometric/unitary
    return psi.isometrize(method='cayley')

def expect_fn(psi, E):
    # compute the total energy, here quimb handles constructing 
    # and contracting all the appropriate lightcones 

    for i in range (L+nb):
        psi = psi&psi_pqc.tensors[i]
    psi_H = psi.H
    for i in range(L):
        psi_H  = psi_H.reindex({f'k{i}':f'b{i}'})
    A = (psi_H &HH& psi)^all
    B = E*(psi_H &H& psi)^all
    C = torch.conj(E)*(psi_H &H_conj& psi)^all
    D = E*torch.conj(E)
    return A-B-C+D

def energy_fn(psi):
    # compute the total energy, here quimb handles constructing 
    # and contracting all the appropriate lightcones 

    for i in range (L+nb):
        psi = psi&psi_pqc.tensors[i]
    psi_H = psi.H
    for i in range(L):
        psi_H  = psi_H.reindex({f'k{i}':f'b{i}'})
    return (psi_H &H& psi)^all
def save_para(qmps_old): #transfer parameters between 2 qmps;
    tag_list=list(qmps_old.tags)
    tag_final=[]
    for i_index in tag_list: 
        if i_index.startswith('G'): tag_final.append(i_index)
    dic_mps={}
    for i in tag_final:
        t = qmps_old[i]
        t = t if isinstance(t, tuple) else [t]
        dic_mps[i] = t[0].data
    return dic_mps



class TNModel_non_hermitian(torch.nn.Module):

    def __init__(self, tn, E):
        super().__init__()
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(tn)
        # n.b. you might want to do extra processing here to e.g. store each
        # parameter as a reshaped matrix (from left_inds -> right_inds), for 
        # some optimizers, and for some torch parametrizations
        self.torch_params = torch.nn.ParameterDict({
            # torch requires strings as keys
            str(i): torch.nn.Parameter(initial)
            for i, initial in params.items()
        })
        indx = str(len(params))
        self.torch_params[indx] = torch.nn.Parameter(E)

    def forward(self):
        # convert back to original int key format
        params = {int(i): p for i, p in list(self.torch_params.items())[:-1]}
        # reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # isometrize and then return the energy
        E = list(self.torch_params.items())[-1][1]
        return expect_fn(norm_fn(psi), E)

class TNModel_energy(torch.nn.Module):

    def __init__(self, tn):
        super().__init__()
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(tn)
        # n.b. you might want to do extra processing here to e.g. store each
        # parameter as a reshaped matrix (from left_inds -> right_inds), for 
        # some optimizers, and for some torch parametrizations
        self.torch_params = torch.nn.ParameterDict({
            # torch requires strings as keys
            str(i): torch.nn.Parameter(initial)
            for i, initial in params.items()
        })

    def forward(self):
        # convert back to original int key format
        params = {int(i): p for i, p in list(self.torch_params.items())}
        # reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # isometrize and then return the energy
        return energy_fn(norm_fn(psi))
ds = [2,4,6,8,10]*4
qs = [2,3,4,5,]*5
ls = zip(ds,qs)

depth, nb = list(ls)[rank]
L = 32
psi_pqc = qmps_f(L+nb, in_depth= depth, n_Qbit=nb, qmps_structure="brickwall", canon="left",)

psi = psi_pqc.tensors[L+nb]
for i in range (L+nb+1,len(psi_pqc.tensors)):
    psi = psi&psi_pqc.tensors[i]
psi_pqc.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
k = 0
Ham = model_mpo.tfim_ising(1,k,1,L) #Jzz, hz, hx, we tune hz
#J, Delta, hz = -1, .5, 0
#Ham = model_mpo.xxz(J, Delta, hz, L)
# boundary conditions
H_bvecl = np.zeros(3)
H_bvecr = np.zeros(3)
H_bvecr[0] = 1.
H_bvecl[-1] = 1.
bdry_vecs2 = [H_bvecl,H_bvecr]

H_tensor = H_contract(Ham,L,H_bvecl,H_bvecr, pbc = False)
H_tensor = H_tensor.reindex({f'p_out{i}':f'k{i}' for i in range(L)})
H_tensor = H_tensor.reindex({f'pc_out{i}':f'b{i}' for i in range(L)})

H = H_tensor

H_conj = H.H
H_conj_1 = H_conj.reindex({f'k{i}':f'bc{i}' for i in range(L)})
H_conj_1 = H_conj_1.reindex({f'b{i}':f'kc{i}' for i in range(L)})
H_1 = H.reindex({f'b{i}':f'kc{i}' for i in range(L)})

HH = H_1&H_conj_1
HH = HH.reindex({f'bc{i}':f'b{i}' for i in range(L)})
H.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
H_conj.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
HH.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))

model = TNModel_energy(psi)
model()

import warnings
from torch import optim
with warnings.catch_warnings():
    warnings.filterwarnings(
        action='ignore',
        message='.*trace might not generalize.*',
    )
    model = torch.jit.trace_module(model, {"forward": []})
    
import torch_optimizer
import tqdm
optimizer = optim.Adam(model.parameters(), lr=.005)

its = 4000
pbar = tqdm.tqdm(range(its),disable=False)

for _ in pbar:
    show_progress_bar=False
    optimizer.zero_grad()
    loss = model()
    loss.backward()
    def closure():
        return loss
    optimizer.step()
    pbar.set_description(f"{loss}")
    progress_bar_refresh_rate=0
dictionary = save_para(psi)
with open(f'results/mps_L{L}_k{"%.2g" % k}_nb{nb}_depth{depth}.pkl', 'wb') as f:
    pickle.dump(dictionary, f)
    
    
Elist = [energy_fn(norm_fn(psi))]
E = Variable(torch.tensor(Elist[0], dtype=torch.complex128), requires_grad=True)

for i in range (1,5):
    #Ham = model_mpo.xxz(-1,.5+0.05j*i,0,L) #Jzz, hz, hx, we tune hz
    k = 0.05*i
    Ham = model_mpo.tfim_ising(1,k*1j,1,L)
    # boundary conditions
    
    H_tensor = H_contract(Ham,L,H_bvecl,H_bvecr)
    H_tensor = H_tensor.reindex({f'p_out{i}':f'k{i}' for i in range(L)})
    H_tensor = H_tensor.reindex({f'pc_out{i}':f'b{i}' for i in range(L)})
    
    H = H_tensor
    
    H_conj = H.H
    H_conj_1 = H_conj.reindex({f'k{i}':f'bc{i}' for i in range(L)})
    H_conj_1 = H_conj_1.reindex({f'b{i}':f'kc{i}' for i in range(L)})
    H_1 = H.reindex({f'b{i}':f'kc{i}' for i in range(L)})
    
    HH = H_1&H_conj_1
    HH = HH.reindex({f'bc{i}':f'b{i}' for i in range(L)})
    H.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
    H_conj.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
    HH.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
    
    model = TNModel_non_hermitian(psi,E)
    model()
    
    import warnings
    from torch import optim
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            message='.*trace might not generalize.*',
        )
        model = torch.jit.trace_module(model, {"forward": []})
        
    import torch_optimizer
    import tqdm
    optimizer = optim.Adam(model.parameters(), lr=.001)
    pbar = tqdm.tqdm(range(its),disable=False)
    
    for _ in pbar:
        show_progress_bar=False
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        def closure():
            return loss
        optimizer.step()
        pbar.set_description(f"{loss}")
        progress_bar_refresh_rate=0
    Elist.append(E.detach().numpy().copy())
    dictionary = save_para(psi)
    with open(f'results/mps_L{L}_k{"%.2g" % k}_nb{nb}_depth{depth}.pkl', 'wb') as f:
        pickle.dump(dictionary, f)

total_E_list =comm.gather(Elist, root = 0)

if rank ==0:
    print(total_E_list)