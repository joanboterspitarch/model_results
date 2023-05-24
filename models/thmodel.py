import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
import torch 
import torch.nn as nn
import torch.nn.functional as F

def enlarge_process(matrix, n, C):

    row = matrix[0, :].numpy()
    col = matrix[:, -1].numpy()
    col = col[::-1]
    size = row.shape[0]
    A = matrix.numpy()

    for i in range(1, n):
        A_aux = np.zeros((size+1, size+1))
        A_aux[-size:, 0:size] = A.copy()

        new_row, new_col = np.zeros(size+1), np.zeros(size+1)

        new_row[:-2], new_row[-1] = row[:-1]/C, row[-1]/C
        new_row[-2] = (new_row[-3] + new_row[-1])/2

        new_col[:-2], new_col[-1] = col[:-1]/C, col[-1]/C
        new_col[-2] = (new_col[-3] + new_col[-1])/2

        
        A_aux[0, :] = new_row.copy()
        A_aux[:, -1][::-1] = new_col.copy()     

        A = A_aux.copy()
        row, col = new_row.copy(), new_col.copy()
        size = row.shape[0]

    return torch.from_numpy(A).float()

class Grid:

    def __init__(self, x=None, y=None, mode='bernoulli'):

        if x is None or y is None:
            raise ValueError('This model needs x and y. Please, provide them.')
        
        self.x = x
        self.y = torch.from_numpy(y).float()
        self.K = self.x.shape[0]
        self.N = self.y.shape[0]
        self.n = int(self.N/2)
        
        # initialize states and other variables

        self.state = self.y[:, :, 0].clone()
        self.ind = torch.argwhere(self.state == 1).numpy()
        self.cont = self.state.clone()
        self.neigh_prob = torch.zeros(size=(self.N, self.N), dtype=torch.float)

        self.susceptible = torch.sum(self.state==0).item()
        self.infected = torch.sum(self.state==1).item()
        self.dead = torch.sum(self.state==2).item()
        self.mode = mode

        # attributes related to the spread process
        self.S = torch.zeros(self.N, self.N, self.K+1, dtype=torch.float)
        self.P = torch.zeros(self.N, self.N, self.K, dtype=torch.float)
        self.S[:, :, 0] = self.state.clone()

        # save columns of the dataframe in attributes
        self.Theta = torch.from_numpy(self.x.Theta.values).float()
        self.Rho = torch.from_numpy(self.x.Rho.values).float()
        self.Temp = torch.from_numpy(self.x.Temperatura.values).float()
        self.Hum = torch.from_numpy(self.x.Humedad.values).float()
        self.Train = torch.from_numpy(self.x.Train.values).float()
    
    def initialize(self, inc=1, part=[0, 0.1, 0.5, 0.9, 1.01]):

        self.inc = inc # delay steps between infected and dead cells.
        self.partition = part # partition of the rho axis (3 values).
        self.n = len(part) - 2

        # variables related to the wind
        self.Q = (self.Theta/(torch.pi/2) + 1).type(torch.int8)
        self.Xi = torch.where(
            torch.logical_or(self.Q == 1, self.Q == 3),
            self.Theta - ((self.Q-1)/2)*torch.pi,
            (self.Q/2)*torch.pi - self.Theta
        )
        
        self.m = torch.zeros_like(self.Rho, dtype=torch.int)
        for i, rho in enumerate(self.Rho):
            cont = -1
            aux = False
            while not aux:
                try:
                    cont += 1
                    a = (part[cont] <= rho) and (rho <= part[cont+1])
                    b = (part[cont+1] <= rho) and (rho <= part[cont+2])
                    if a and not b:
                        aux = True
                        self.m[i] = cont
                except IndexError:
                    self.m[i] = self.n + 1
        
        # create dataframes to save data
        self.df_wind = pd.DataFrame(
            {
                'Theta': self.Theta,
                'Rho': self.Rho,
                'Q': self.Q,
                'Xi': self.Xi,
                'm': self.m
            }
        )
        self.df_spread = pd.DataFrame(
            index=range(self.K+1),
            columns=['Susceptible', 'Infected', 'Dead']
        )
        self.df_spread.iloc[0] = [self.susceptible, self.infected, self.dead]

    def compute_th_param(self, alpha=1., beta=1., gamma=1., t_min=0, grad=False):

        self.alpha = torch.tensor(alpha, requires_grad=grad).float()
        self.beta = torch.tensor(beta, requires_grad=grad).float()
        self.gamma = torch.tensor(gamma, requires_grad=grad).float()

        f = (self.Temp - t_min)**self.alpha
        g = (self.Hum)**self.beta

        self.div = 1 + self.gamma * (g/f)
        self.p0 = 1 / self.div

    def submatrix(self):

        sin = torch.sin(self.Xi)
        cos = torch.cos(self.Xi)
        f = torch.where(
            self.Xi <= torch.pi/4,
            torch.tan(self.Xi),
            1/torch.tan(self.Xi)
        )
        self.A = torch.zeros(2, 2, self.K, dtype=torch.float)
        self.A[0, 0, :] = sin
        self.A[0, 1, :] = f
        self.A[1, 0, :] = 0
        self.A[1, 1, :] = cos

    def enlargement_process(self):
        
        self.large_matrices = torch.zeros(2*self.n+1, 2*self.n+1, self.K, dtype=torch.float)

        # when rho <= part[0]
        ind = torch.argwhere(self.m == 0)
        for i in ind:
            self.large_matrices[(self.n-1):(self.n+2), (self.n-1):(self.n+2), ind] = self.p0[i]
            self.large_matrices[self.n, self.n, ind] = 0

        # ENLARGEMENT PROCESS

        ind = torch.argwhere(self.m != 0)

        for i in ind:
            m = self.m[i]
            A = self.A[:, :, i].reshape(2, 2).clone()
            new_A = enlarge_process(matrix=A, n=m, C=self.div[i])
            size = new_A.shape[0]
            self.large_matrices[(self.n-size+1):(self.n+1), (self.n):(self.n+size), i] = new_A.reshape(size, size, 1)

        # rotate the matrices to the right position
        q2 = (self.Q == 2)
        q3 = (self.Q == 3)
        q4 = (self.Q == 4)

        self.large_matrices[:, :, q2] = torch.transpose(
            torch.rot90(
                self.large_matrices[:, :, q2],
                k=1
            ),
            0,
            1
        )

        self.large_matrices[:, :, q4] = torch.transpose(
            torch.rot90(
                self.large_matrices[:, :, q4],
                k=-1
            ),
            0,
            1
        )

        self.large_matrices[:, :, q3] = torch.rot90(
            self.large_matrices[:, :, q3],
            k = 2
        )

    def neighbourhood_relation(self, step):

        padding = torch.zeros(self.N + 2*self.n, self.N + 2*self.n, dtype=torch.float)
        for i,j in self.ind:
            padding[i:(i+(2*self.n+1)), j:(j+(2*self.n+1))] += self.large_matrices[:, :, step]
        self.neigh_prob = padding[self.n:-self.n, self.n:-self.n].clone()

    def update(self, tau=1):

        infected_cells = (self.state == 1)
        # the new dead cells are infected cells which have already completed the delay
        ind_new_dead_cells = torch.logical_and(infected_cells, self.cont == self.inc)
        self.state[ind_new_dead_cells] += 1 # infected to dead once verified the delay, we use += due to backpropagation
        # update the state of the cells
        # 1. for every single healthy cell, whose probability of being infected is 0 or higher than one
        #    the cell becomes healthy or infected respectively.
        health_cells = (self.state == 0)
        #self.state[torch.logical_and(self.neigh_prob <= 0, health_cells)] = 0
        self.state[torch.logical_and(self.neigh_prob >= 1, health_cells)] += 1
        # 2. for every single healthy cell, whose probability of being infected is between 0 and 1
        #    healthy cell + prob between 0, 1 --> update the state of the cell
        to_inf_cells = torch.logical_and(self.neigh_prob < 1, self.neigh_prob > 0)
        ind_to_inf_cells = torch.logical_and(to_inf_cells, health_cells)
        if self.mode == 'bernoulli':
            self.state[ind_to_inf_cells] = torch.distributions.Bernoulli(self.neigh_prob[ind_to_inf_cells]).sample().to(dtype=torch.float)    
        elif self.mode == 'gumbel':
            probs = torch.stack(
                (self.neigh_prob[ind_to_inf_cells], 1 - self.neigh_prob[ind_to_inf_cells]),
                dim=1
            ).log()
            self.state[ind_to_inf_cells] = F.gumbel_softmax(logits=probs, tau=tau, hard=True)[:, 0].to(dtype=torch.float)
        # it is necessary to update the counter of the infected cells
        self.cont[self.state==1] += 1
        # update the indices of the infected cells
        self.ind = torch.argwhere(self.state==1).numpy()
        # update the number of susceptible, infected and dead cells
        self.susceptible = torch.sum(self.state==0).item()
        self.infected = torch.sum(self.state==1).item()
        self.dead = torch.sum(self.state==2).item()

    def spread(self, seed=0, tau=1):

        torch.random.manual_seed(seed)
        np.random.seed(seed)

        for L in range(self.K):
            self.neighbourhood_relation(step=L)
            self.P[:, :, L] = self.neigh_prob.clone()
            self.update(tau=tau)
            self.S[:, :, L+1] = self.state.clone()
            self.df_spread.iloc[L+1] = [self.susceptible, self.infected, self.dead]
    
    def refresh(self):

        self.state = self.y[:, :, 0].clone()
        self.cont = self.state.clone()
        self.ind = torch.argwhere(self.state==1).numpy()
        self.neigh_prob = torch.zeros(size=(self.N, self.N), dtype=torch.float)

        self.susceptible = torch.sum(self.state==0).item()
        self.infected = torch.sum(self.state==1).item()
        self.dead = torch.sum(self.state==2).item()
            
        self.S = torch.zeros(self.N, self.N, self.K+1, dtype=torch.float)
        self.S[:, :, 0] = self.state.clone()
        self.P = torch.zeros(self.N, self.N, self.K, dtype=torch.float)

        self.df_spread = pd.DataFrame(
            np.zeros((self.K+1, 3), dtype='float64'),
            columns=['Susceptible', 'Infected', 'Dead']
        )
        self.df_spread.iloc[0] = [self.susceptible, self.infected, self.dead]

    def task_montecarlo(self, n_it=3*10**2, tau=1):

        for s in range(n_it):

            torch.random.manual_seed(s)
            np.random.seed(s)

            self.refresh()
            self.spread(seed=s, tau=tau)
            
            self.X0 += torch.where(
                self.S==0,
                self.S + 1,
                0
            ).clone().to(torch.float)

            self.X1 += torch.where(
                self.S==1,
                self.S,
                0
            ).clone().to(torch.float)

            self.X2 += torch.where(
                self.S==2,
                self.S - 1,
                0
            ).clone().to(torch.float)

            self.P_MC += self.P
            self.df_MC += self.df_spread
        
        self.X0 /= n_it
        self.X1 /= n_it
        self.X2 /= n_it
        self.P_MC /= n_it
        self.df_MC /= n_it

    def montecarlo(self, n_it=10**3, tau=1):

        # we create our tensors to storage the results

        self.X0 = torch.zeros(self.N, self.N, self.K+1, dtype=torch.float)
        self.X1 = torch.zeros(self.N, self.N, self.K+1, dtype=torch.float)
        self.X2 = torch.zeros(self.N, self.N, self.K+1, dtype=torch.float)
        self.P_MC = torch.zeros(self.N, self.N, self.K, dtype=torch.float)
        self.df_MC = pd.DataFrame(
            np.zeros((self.K+1, 3), dtype='float64'),
            columns=['Susceptible', 'Infected', 'Dead']
        )

        #self.submatrix()
        #self.enlargement_process()

        # we have to note that self.A and self.large_matrices are already computed.
        # we only need to compute the neighbourhood relation and update the state
        # for every single random seed

        self.task_montecarlo(n_it=n_it, tau=tau)

        _, self.X = torch.max(
            torch.stack((self.X0, self.X1, self.X2), dim=-1),
            dim=-1
        )
    

def Train(x, y, alpha=1., beta=1., gamma=1., delta=0.95, a1=1, a2=3, inc=1, part=[0.1, 0.5, 0.9], n_it=100, epochs=10):

    torch.random.manual_seed(0)
    np.random.seed(0)
    
    x.Train.values[0] = False
    indices = np.argwhere(x.Train.values==True).flatten()
    target = torch.from_numpy(y[:, :, 1:]).to(dtype=torch.float)
    target = torch.where(
        target==0,
        0,
        1
    ).float()

    l1 = nn.MSELoss(reduction='sum')
    l2 = nn.BCELoss()

    for epoch in range(epochs):

        print('Epoch: ', epoch)

        thmodel = Grid(x=x, y=y, mode='gumbel')
        thmodel.initialize(inc=inc, part=part)
        thmodel.compute_th_param(alpha=alpha, beta=beta, gamma=gamma, grad=True)
        thmodel.submatrix()
        thmodel.enlargement_process()
        thmodel.montecarlo(n_it=n_it)

        #optimizer = torch.optim.Adam([thmodel.alpha, thmodel.beta, thmodel.gamma], lr=0.15)

        loss = 0
        #optimizer.zero_grad()
        r = 0
        for ind in indices:
            probs = (thmodel.X1[:, :, ind] + thmodel.X2[:, :, ind]).flatten().float()
            #l_1 = a1*l1(probs, target[:, :, r].flatten())
            #l_2 = a2*l2(probs, target[:, :, r].flatten())
            #loss += (delta**r)*(l_1 + l_2)
            loss += (gamma**r)*(l2(probs, target[:, :, r].flatten()) + l1(probs, target[:, :, r].flatten()))
            r += 1
        # normalize the gradients
        

        loss.backward()
        #optimizer.step()
        with torch.no_grad():
            alpha += 0.1 * thmodel.alpha.grad / (thmodel.alpha.grad.norm() + 1e-8)
            beta += 0.1 * thmodel.beta.grad / (thmodel.beta.grad.norm() + 1e-8)
            gamma += 0.1 * thmodel.gamma.grad / (thmodel.gamma.grad.norm() + 1e-8)

        print('Loss: ', loss.item())
        print('Alpha: ', alpha.item(), 'Beta: ', beta.item(), 'Gamma: ', gamma.item())
        #print('Gradientes: ', thmodel.alpha.grad.item(), thmodel.beta.grad.item(), thmodel.gamma.grad.item())
    
    return thmodel, alpha, beta, gamma




        