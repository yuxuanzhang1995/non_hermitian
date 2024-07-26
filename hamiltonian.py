

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Shahin, Yuxuan
"""

import numpy as np
class model_mpo(object): 
    """
    Matrix product operator (MPO) representation of
    model Hamiltonian (written in the thermal states
    convention).
    tensor leg index notation: p_out, b_out, p_in, b_in
    N = number of sites
    """ 
    def tfim_ising(J, g, h, N):
        """
        Unit-cell matrix product operator of Transverse 
        Field Ising model (TFIM). 
        """
        # Pauli matrices 
        sigmax = np.array([[0., 1], [1, 0.]])
        sigmay = np.array([[0., -1j], [1j, 0.]])
        sigmaz = np.array([[1, 0.], [0., -1]])
        id = np.eye(2)
    
       # structure of TFIM Ising model MPO unit cell
        H = np.zeros((3, 3, 2, 2), dtype=np.complex128)+0j
        H[0, 0] = H[2, 2] = id
        H[1, 0] = sigmaz
        H[2, 0] = (-g * sigmaz - h * sigmax)#/N
        H[2, 1] = (-J * sigmaz)#/(N-1)
        H1 = np.einsum('abcd->cbda',H)
        return H1 
    
    def sd_ising(J, V, h, N):
        """
        Unit-cell matrix product operator of self-dual 
        Ising model. 
        """
        # Pauli matrices
        sigmax = np.array([[0., 1], [1, 0.]])
        sigmay = np.array([[0., -1j], [1j, 0.]])
        sigmaz = np.array([[1, 0.], [0., -1]])
        id = np.eye(2)
    
        # structure of self-dual Ising model MPO unit cell
        H = np.zeros((5, 5, 2, 2), dtype=np.float)
        H[0, 0] = H[3, 2] = H[4, 4] = id
        H[1, 0] = sigmax
        H[2, 0] = sigmaz
        H[4, 0] = (-h * sigmax)#/N
        H[4, 1] = (V * sigmax)#/(N - 1)
        H[4, 2] = (-J * sigmaz)#/(N - 1)
        H[4, 3] = (V * sigmaz)#/(N - 2)
        H1 = np.einsum('abcd->cbda',H)
        return H1
    
    def xxz(J, Delta, hz, N):
        """
        Unit-cell matrix product operator of anisotropic 
        Heisenberg XXZ chain model
        """
        # Pauli matrices 
        sigmax = np.array([[0., 1], [1, 0.]])
        sigmay = np.array([[0., -1j], [1j, 0.]])
        sigmaz = np.array([[1, 0.], [0., -1]])
        id = np.eye(2)
    
        # structure of XXZ model MPO unit cell
        H = np.zeros((5, 5, 2, 2), dtype=np.complex128) 
        H[0, 0] = H[4, 4] = id
        H[1, 0] = sigmax
        H[2, 0] = sigmay
        H[3, 0] = sigmaz
        H[4, 0] = (hz * sigmaz)#/N
        H[4, 1] = (J * sigmax)#/(N - 1)
        H[4, 2] = (J * sigmay)#/(N - 1)
        H[4, 3] = (J * Delta * sigmaz)#/(N - 1)
        H1 = np.einsum('abcd->cbda',H)
        return H1+0*1j
    
    def fermi_hubbard(t_val, U_val, mu_val, N):
        """
         Unit-cell matrix product operator of 
         spinhalf fermion Fermi-Hubbard model
        """
        t, U, mu = t_val/(N-1), U_val/N, mu_val/N
        H = np.zeros((6, 6, 4, 4), dtype=np.float)
        H[0, 0] = H[5, 5] = np.eye(4)
    
        H[1,0] = np.array([[0.,0.,-1.,0.],
                           [0.,0.,0.,-1.],
                           [0.,0.,0.,0.],
                           [0.,0.,0.,0.]])
    
        H[2,0] = np.array([[0.,0.,0.,0.],
                           [0.,0.,0.,0.],
                           [1.,0.,0.,0.],
                           [0.,1.,0.,0.]])
    
        H[3,0] = np.array([[0.,0.,0.,0.],
                           [1.,0.,0.,0.],
                           [0.,0.,0.,0.],
                           [0.,0.,-1.,0.]])
    
        H[4,0] = np.array([[0.,-1.,0.,0.],
                           [0.,0.,0.,0.],
                           [0.,0.,0.,1.],
                           [0.,0.,0.,0.]])
    
        H[5,0] = np.array([[0.,0.,0.,0.],
                           [0.,-mu,0.,0.],
                           [0.,0.,-mu,0.],
                           [0.,0.,0.,-2*mu + U]])
    
        H[5,1] = np.array([[0.,0.,0.,0.],
                           [0.,0.,0.,0.],
                           [t,0.,0.,0.],
                           [0.,-t,0.,0.]])
    
        H[5,2] = np.array([[0.,0.,-t,0.],
                           [0.,0.,0.,t],
                           [0.,0.,0.,0.],
                           [0.,0.,0.,0.]])
    
        H[5,3] = np.array([[0.,-t,0.,0.],
                           [0.,0.,0.,0.],
                           [0.,0.,0.,-t],
                           [0.,0.,0.,0.]])
    
        H[5,4] = np.array([[0.,0.,0.,0.],
                           [t,0.,0.,0.],
                           [0.,0.,0.,0.],
                           [0.,0.,t,0.]])
        H1 = np.einsum('abcd->cbda',H)
        return H1
    
    def xxz2(J, Delta, hz, N):
        """
        Unit-cell matrix product operator of anisotropic 
        Heisenberg XXZ chain model (version 2)
        """
        # Pauli matrices 
        sigmax = np.array([[0., 1], [1, 0.]])
        sigmay = np.array([[0., -1j], [1j, 0.]])
        sigmaz = np.array([[1, 0.], [0., -1]])
        sigmap = sigmax + 1j*sigmay
        sigman = sigmax - 1j*sigmay
        id = np.eye(2)
    
        # structure of XXZ model MPO unit cell
        H = np.zeros((5, 5, 2, 2), dtype=np.complex) 
        H[0, 0] = H[4, 4] = id
        H[1, 0] = sigmap
        H[2, 0] = sigman
        H[3, 0] = sigmaz
        H[4, 0] = (hz * sigmaz)/N
        H[4, 1] = (0.5* J * sigman)/(N - 1)
        H[4, 2] = (0.5* J * sigmap)/(N - 1)
        H[4, 3] = (J * Delta * sigmaz)/(N - 1)
        H1 = np.einsum('abcd->cbda',H)
        return H1
