import torch
import numpy as np

class IntegrationFext:
    def __init__(self, dim):
        self.dim = dim

    def lossFextEnergy(self, u,x, neuBC_coordinates, neuBC_values, neuBC_idx, dxdydz):
        dx=dxdydz[0]
        dy=dxdydz[1]
        dxds=dx/2
        dydt=dy/2
        
        # J= np.array([[dxds,0],[0,dydt]])
        # Jinv= np.linalg.inv(J)
        # detJ= np.linalg.det(J)
        
        traction_ID = 0
        neuPt_u = u[ neuBC_idx[traction_ID].cpu().numpy() ]
        W_ext = torch.einsum( 'ij,ij->i' , neuPt_u , neuBC_values[traction_ID] ) * dx

        W_ext[-1] = W_ext[-1] / 2
        W_ext[0] = W_ext[0] / 2
        
        FextEnergy = torch.sum( W_ext )
        return FextEnergy