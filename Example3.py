# -*- coding: utf-8 -*-
"""
Created on Sat May 28 20:11:11 2022

@author: charul
"""

# from Sub_Functions import define_structure as des
from Sub_Functions.MultiLayerNet import *
from Sub_Functions.InternalEnergy import *
from Sub_Functions.IntegrationFext import *
from Sub_Functions import Utility as util
from Sub_Functions.MMA import *
from torch.autograd import grad
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
import time
import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy.random as npr
import random
import os
from sklearn.preprocessing import normalize
from scipy.sparse import *
from skimage.morphology import disk
import warnings
warnings.filterwarnings("ignore")
npr.seed(2022)
torch.manual_seed(2022)
np.random.seed(2022)

# torch.set_default_dtype(torch.float64)
# torch.set_default_tensor_type(torch.DoubleTensor)

# torch.cuda.is_available = lambda : False
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    device_string = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device_string = 'cpu'
    dev = torch.device('cpu')
    print("CUDA not available, running on CPU")

##################################################  DEM functions  ##########################################################################
def get_Train_domain():
    x_dom = x_min, Length, Nx
    y_dom = y_min, Height, Ny
    # create points
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
    dom = np.zeros((Nx * Ny, 2)) #Initializing the domain array
    c = 0
    
    node_dy= (y_dom[1]-y_dom[0])/(y_dom[2]-1)
    node_dx= (x_dom[1]-x_dom[0])/(x_dom[2]-1)
    
    # Assign nodal coordinates to all the points in the dom array
    for x in np.nditer(lin_x):
        tb = y_dom[2] * c
        te = tb + y_dom[2]
        c += 1
        dom[tb:te, 0] = x
        dom[tb:te, 1] = lin_y
    
    # Plot the points defined in Dom
    np.meshgrid(lin_x, lin_y)
    # fig = plt.figure(figsize=(5, 1))
    # ax = fig.add_subplot(111)
    # ax.scatter(dom[:, 0], dom[:, 1], s=0.005, facecolor='blue')
    # ax.set_xlabel('X', fontsize=3)
    # ax.set_ylabel('Y', fontsize=3)
    # ax.tick_params(labelsize=4)
    return dom, {} , {}

def get_Test_datatest(Nx, Ny):
    x_dom_test = x_min, Length_test, Nx
    y_dom_test = y_min, Height_test, Ny
    # create points
    x_space = np.linspace(x_dom_test[0], x_dom_test[1], x_dom_test[2])
    y_space = np.linspace(y_dom_test[0], y_dom_test[1], y_dom_test[2])
    xGrid, yGrid = np.meshgrid(x_space, y_space)
    data_test = np.concatenate(
        (np.array([xGrid.flatten()]).T, np.array([yGrid.flatten()]).T), axis=1)
    return x_space, y_space, data_test

def ConvergenceCheck( arry , rel_tol ):
    num_check = 10

    # Run minimum of 2*num_check iterations
    if len( arry ) < 2 * num_check :
        return False

    mean1 = np.mean( arry[ -2*num_check : -num_check ] )
    mean2 = np.mean( arry[ -num_check : ] )

    if np.abs( mean2 ) < 1e-6:
        print('Loss value converged to abs tol of 1e-6' )
        return True     

    if ( np.abs( mean1 - mean2 ) / np.abs( mean2 ) ) < rel_tol:
        print('Loss value converged to rel tol of ' + str(rel_tol) )
        return True
    else:
        return False
   
class DeepEnergyMethod:
    # Instance attributes
    def __init__(self, model, dim, E, nu, act_func, CNN_dev, rff_dev,N_Layers):
        # Model 1
        self.model = MultiLayerNet(model[0], model[1], model[2],act_func, CNN_dev, rff_dev,N_Layers)
        self.model = self.model.to(dev)
        
        # Model 2
        self.model2 = MultiLayerNet(model[0], model[1], model[2],act_func, CNN_dev, rff_dev,N_Layers)
        self.model2 = self.model2.to(dev)

        self.InternalEnergy= InternalEnergy(E, nu)
        self.dim = dim
        self.lossArray = []

    def PBC_Loss( self , u_pred , mode , Nx , Ny ):
        ux = u_pred[:,0].reshape( [Ny,Nx] ).T
        uy = u_pred[:,1].reshape( [Ny,Nx] ).T
        wt = E * 10

        if mode == 1:
            loss_PBC_x = torch.sum( torch.pow( ux[0,1:-1] - ux[-1,1:-1] , 2. ) ) * wt
            loss_PBC_y = torch.sum( torch.pow( uy[0,1:-1] - uy[-1,1:-1] , 2. ) ) * wt + torch.sum( torch.pow( uy[1:-1,0] - uy[1:-1,-1] , 2. ) ) * wt 

        if mode == 2:
            loss_PBC_x = torch.sum( torch.pow( ux[1:-1,0] - ux[1:-1,-1] , 2. ) ) * wt + torch.sum( torch.pow( ux[0,1:-1] - ux[-1,1:-1] , 2. ) ) * wt
            loss_PBC_y = torch.sum( torch.pow( uy[1:-1,0] - uy[1:-1,-1] , 2. ) ) * wt

        if mode == 3:
            loss_PBC_x = torch.sum( torch.pow( ux[1:-1,0] - ux[1:-1,-1] , 2. ) ) * wt
            loss_PBC_y = torch.sum( torch.pow( uy[0,1:-1] - uy[-1,1:-1] , 2. ) ) * wt + torch.sum( torch.pow( uy[1:-1,0] - uy[1:-1,-1] , 2. ) ) * wt 

        return loss_PBC_x , loss_PBC_y



    def train_model(self, shape, dxdydz, data, neumannBC, dirichletBC, iteration, learning_rate,N_Layers,activatn_fn, rho , TO_itr ):            
        global Train_time , ObjVal
        x = torch.from_numpy(data).float()
        x = x.to(dev)
        x.requires_grad_(True)

        density = torch.from_numpy(rho).float()
        density = torch.reshape( density , [ Ny-1 , Nx-1 ] ).to(dev)

        # ----------------------------------------------------------------------------------
        # Minimizing loss function (energy and boundary conditions)
        # ----------------------------------------------------------------------------------
        start_time = time.time()

        # Train first model
        if Example == 'Shear':
            mode = 3
        elif Example == 'Bulk':
            mode = 2
        optimizer_LBFGS = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=20,line_search_fn='strong_wolfe')
        MOD = self.model
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                it_time = time.time()
                u_pred = self.getU(x,N_Layers,activatn_fn, mode , MOD )
                u_pred.double()

                # ---- Calculate internal and external energies------                 
                loss_int = self.InternalEnergy.Elastic2DGaussQuadMeta( u_pred , None , x, dxdydz, shape, density , False , Example , TO_itr )
                loss_PBC_x , loss_PBC_y = self.PBC_Loss( u_pred , mode , Nx , Ny )
                loss = loss_int + loss_PBC_x + loss_PBC_y
                # loss = loss_int
                optimizer_LBFGS.zero_grad()
                loss.backward()

                if verbose:
                    print('     Iter: %d Loss: %.6e PBCx: %.4e PBCy: %.4e'% (t + 1, loss.item(), loss_PBC_x.item(),loss_PBC_y.item() ))
                    # print('     Iter: %d Loss: %.6e '% (t + 1, loss.item() ))
                self.lossArray.append( loss.data.cpu() )
                return loss
            optimizer_LBFGS.step(closure)

            # Check convergence
            if ConvergenceCheck( self.lossArray , convergence_tol[ TO_itr ] ):
                break
        elapsed = time.time() - start_time
        print('Model 1 training time: %.4f' % elapsed)
        Train_time.append( elapsed )

        if Example != 'Shear':
            mode = 1
            optimizer_LBFGS2 = torch.optim.LBFGS(self.model2.parameters(), lr=learning_rate, max_iter=20,line_search_fn='strong_wolfe')
            MOD = self.model2
            self.lossArray = []
            for t in range(iteration):
                # Zero gradients, perform a backward pass, and update the weights.
                def closure():
                    it_time = time.time()
                    u_pred = self.getU(x,N_Layers,activatn_fn, mode , MOD )
                    u_pred.double()

                    # ---- Calculate internal and external energies------                 
                    loss_int = self.InternalEnergy.Elastic2DGaussQuadMeta( u_pred , None , x, dxdydz, shape, density , False , Example , TO_itr )
                    loss_PBC_x , loss_PBC_y = self.PBC_Loss( u_pred , mode , Nx , Ny )
                    loss = loss_int + loss_PBC_x + loss_PBC_y
                    # loss = loss_int

                    optimizer_LBFGS2.zero_grad()
                    loss.backward()

                    if verbose:
                        print('     Iter: %d Loss: %.6e PBCx: %.4e PBCy: %.4e'% (t + 1, loss.item(), loss_PBC_x.item(),loss_PBC_y.item() ))
                        # print('     Iter: %d Loss: %.6e '% (t + 1, loss.item() ))
                    self.lossArray.append( loss.data.cpu() )
                    return loss
                optimizer_LBFGS2.step(closure)

                # Check convergence
                if ConvergenceCheck( self.lossArray , convergence_tol[ TO_itr ] ):
                    break
            elapsed = time.time() - start_time
            print('Model 2 training time: %.4f' % elapsed)
            Train_time[-1] += elapsed


        # TO:
        if Example == 'Shear':
            u1 = self.getU(x,N_Layers,activatn_fn, 3 , self.model )
            u2 = None 

        elif Example == 'Bulk':
            u1 = self.getU(x,N_Layers,activatn_fn, 2 , self.model )
            u2 = self.getU(x,N_Layers,activatn_fn, 1 , self.model2 ) 

        #                                                                                                   Sensitivity mode
        dfdrho , compliance = self.InternalEnergy.Elastic2DGaussQuadMeta( u1 , u2 , x, dxdydz, shape, density , True ,            Example , TO_itr )
        ObjVal.append( compliance.cpu().detach().numpy() )
        return dfdrho , compliance


    def getU(self, x,N_Layers,activatn_fn , mode , MOD ):
        u = MOD( x , N_Layers,activatn_fn)
        phix = x[:, 0] / Length
        phiy = x[:, 1] / Height

        if mode == 1:
            # Uniaxial strain in X direction
            Ux = phix * ( 1. - phix ) * u[:, 0] + phix * 0.01 * Length
            Uy = ( ( 1. - phix ) * phix + ( 1. - phiy ) * phiy ) * u[:, 1]

        elif mode == 2:
            # Uniaxial strain in Y direction
            Ux = ( ( 1. - phix ) * phix + ( 1. - phiy ) * phiy ) * u[:, 0]
            Uy = phiy * ( 1. - phiy ) * u[:, 1] + phiy * 0.01 * Height

        elif mode == 3:
            # Shear
            Ux = phiy * ( 1. - phiy ) * u[:, 0] + phiy * 0.01 * Length
            Uy = ( ( 1. - phix ) * phix + ( 1. - phiy ) * phiy ) * u[:, 1]

        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        u_pred = torch.cat((Ux, Uy), -1)
        return u_pred
    

    def evaluate_model(self, x, y, E, nu, N_Layers,activatn_fn , density ):   
        Nx = len(x)
        Ny = len(y)
        xGrid, yGrid = np.meshgrid(x, y)
        x1D = xGrid.flatten()
        y1D = yGrid.flatten()
        xy = np.concatenate((np.array([x1D]).T, np.array([y1D]).T), axis=-1)
        xy_tensor = torch.from_numpy(xy).float()
        xy_tensor = xy_tensor.to(dev)
        xy_tensor.requires_grad_(True)

        if Example == 'Shear':
            mode = 3
            u_pred_torch = self.getU(xy_tensor,N_Layers,activatn_fn , mode , self.model )
        elif Example == 'Bulk':
            mode = 2
            u_pred_torch = self.getU(xy_tensor,N_Layers,activatn_fn , mode , self.model )

        duxdxy = \
            grad(u_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                 create_graph=True, retain_graph=True)[0]
        duydxy = \
            grad(u_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                 create_graph=True, retain_graph=True)[0]

        E11 = duxdxy[:, 0].unsqueeze(1)
        E22 = duydxy[:, 1].unsqueeze(1)
        E12 = (duxdxy[:, 1].unsqueeze(1) + duydxy[:, 0].unsqueeze(1))/2

        S11 = E/(1-nu**2)*(E11 + nu*E22)
        S22 = E/(1-nu**2)*(E22 + nu*E11)
        S12 = E*E12/(1+nu)

        u_pred = u_pred_torch.detach().cpu().numpy()

        flag = np.ones( [Nx*Ny,1] )
        threshold = np.max( density ) * 0.6
        mask = ( density.flatten() < threshold )
        flag[ mask , 0 ] = np.nan

        E11_pred = E11.detach().cpu().numpy() * flag
        E12_pred = E12.detach().cpu().numpy() * flag
        E22_pred = E22.detach().cpu().numpy() * flag
        S11_pred = S11.detach().cpu().numpy() * flag
        S12_pred = S12.detach().cpu().numpy() * flag
        S22_pred = S22.detach().cpu().numpy() * flag

        u_pred[:,0] *= flag[:,0]
        u_pred[:,1] *= flag[:,0]
        surUx = u_pred[:, 0].reshape(Ny, Nx)
        surUy = u_pred[:, 1].reshape(Ny, Nx)
        surUz = np.zeros([Nx, Ny])

        E11 = E11_pred.reshape(Ny, Nx)
        E12 = E12_pred.reshape(Ny, Nx)
        E22 = E22_pred.reshape(Ny, Nx)
        S11 = S11_pred.reshape(Ny, Nx)
        S12 = S12_pred.reshape(Ny, Nx)
        S22 = S22_pred.reshape(Ny, Nx)

        SVonMises = np.float64(
            np.sqrt(0.5 * ((S11 - S22) ** 2 + (S22) ** 2 + (-S11) ** 2 + 6 * (S12 ** 2))))
        U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))

        # Write output
        Write_file = True
        if Write_file:
            z = np.array([0]).astype(np.float64)
            util.write_vtk_v2( filename_out + 'TO_itr_' + str(TO_itr) , x, y, z, U, S11, S12, S22, E11, E12, E22, SVonMises )

        Write_fig = True
        if Write_fig:
            fig, axs = plt.subplots(3, 3, sharex='col')
            ax = axs[0,0]
            ff = ax.imshow( np.flipud( U[0][:,:] ) , extent = [x_min, Length, y_min, Height],cmap='jet')
            plt.colorbar( ff , ax=ax )
            ax.set_aspect('equal') 
            ax.set_title('Ux')

            ax = axs[0,1]
            ff = ax.imshow( np.flipud( U[1][:,:] ) , extent = [x_min, Length, y_min, Height],cmap='jet')
            plt.colorbar( ff , ax=ax )
            ax.set_aspect('equal') 
            ax.set_title('Uy')

            ax = axs[0,2]
            ff = ax.imshow( np.flipud( SVonMises[:,:] ) , extent = [x_min, Length, y_min, Height],cmap='jet')
            plt.colorbar( ff , ax=ax )
            ax.set_aspect('equal') 
            ax.set_title('Mises stress')



            ax = axs[1,0]
            ff = ax.imshow( np.flipud( E11[:,:] ) , extent = [x_min, Length, y_min, Height],cmap='jet')
            plt.colorbar( ff , ax=ax )
            ax.set_aspect('equal') 
            ax.set_title('E11')

            ax = axs[1,1]
            ff = ax.imshow( np.flipud( E22[:,:] ) , extent = [x_min, Length, y_min, Height],cmap='jet')
            plt.colorbar( ff , ax=ax )
            ax.set_aspect('equal') 
            ax.set_title('E22')

            ax = axs[1,2]
            ff = ax.imshow( np.flipud( E12[:,:] ) , extent = [x_min, Length, y_min, Height],cmap='jet')
            plt.colorbar( ff , ax=ax )
            ax.set_aspect('equal') 
            ax.set_title('E12')



            ax = axs[2,0]
            ff = ax.imshow( np.flipud( S11[:,:] ) , extent = [x_min, Length, y_min, Height],cmap='jet')
            plt.colorbar( ff , ax=ax )
            ax.set_aspect('equal') 
            ax.set_title('S11')

            ax = axs[2,1]
            ff = ax.imshow( np.flipud( S22[:,:] ) , extent = [x_min, Length, y_min, Height],cmap='jet')
            plt.colorbar( ff , ax=ax )
            ax.set_aspect('equal') 
            ax.set_title('S22')

            ax = axs[2,2]
            ff = ax.imshow( np.flipud( S12[:,:] ) , extent = [x_min, Length, y_min, Height],cmap='jet')
            plt.colorbar( ff , ax=ax )
            ax.set_aspect('equal') 
            ax.set_title('S12')
            plt.tight_layout()

            plt.savefig( './Example' + str(Example) + '/FieldVars_' + str(TO_itr) + '.png' , format = 'png', dpi=1200 , bbox_inches="tight" )
            plt.close()

        np.save('./Example' + str(Example) + '/ITR_' + str(TO_itr) + '.npy' , np.array([density,\
            U, S11, S12, S22, E11, E12, E22, SVonMises],dtype=object) )


##################################################  TopOpt functions  ##########################################################################
def Filter( rad ):
    nex = Nx-1
    ney = Ny-1
    Lx = Length
    Ly = Height
    dx = Lx / nex
    dy = Ly / ney

    xx = np.linspace(0,Lx,nex)
    yy = np.linspace(0,Ly,ney)
    X , Y = np.meshgrid( xx , yy )
    X = X.flatten()
    Y = Y.flatten()

    wi , wj , wv = [] , [] , []
    for eid in range( nex * ney ):
        my_X = X[eid]
        my_Y = Y[eid]

        dist = np.sqrt( ( X - my_X )**2 + ( Y - my_Y )**2 )
        neighbours = np.where( dist <= rad )[0]
        wi += [eid] * len(neighbours)
        wj += list( neighbours )
        wv += list( rad - dist[ neighbours ] )
    
    W = normalize( coo_matrix( (wv, (wi, wj)), shape=(nex*ney, nex*ney)) , norm='l1', axis=1).tocsr() # Normalize row-wise
    return W

def Val_and_Grad( density ):
    global TO_itr
    dxdy = [hx, hy]
    rho_tilda = W @ density 

    dfdrho_tilda , compliance = dem.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet, iteration, lr,N_Layers,act_func,rho_tilda , TO_itr )
    # Scale to obtain true sensitivity
    dfdrho_tilda /= ( Length * Height )
    compliance /= ( Length * Height )

    dfdrho_tilda_npy = dfdrho_tilda.cpu().detach().numpy()

    # Invert filter
    ss = dfdrho_tilda_npy.shape
    sensitivity = W.T @ dfdrho_tilda_npy.flatten()

    # Save plot
    fig, axs = plt.subplots(2, 1, sharex='col')
    ff = axs[0].imshow( np.flipud( density.reshape(ss) ) , extent = [x_min, Length, y_min, Height],cmap='binary')
    plt.colorbar( ff , ax=axs[0] )
    axs[0].set_aspect('equal') 
    axs[0].set_title('Element density')
    ff = axs[1].imshow( np.flipud( sensitivity.reshape(ss) ) , extent = [x_min, Length, y_min, Height],cmap='jet')
    plt.colorbar( ff , ax=axs[1] )
    axs[1].set_aspect('equal') 
    axs[1].set_title('Sensitivity' )
    plt.savefig( './Example' + str(Example) + '/Design_' + str(TO_itr) + '.png' , format = 'png', dpi=600 , bbox_inches="tight" )
    plt.close()
    
    # Save some data
    dem.evaluate_model(x, y,E, nu,N_Layers,act_func , density.reshape(ss) )

    # exit()

    TO_itr += 1
    return compliance.cpu().detach().numpy() , sensitivity


# Volume constraint
def VolCon( rho ):
    return np.mean( rho ) - Vf , np.ones_like( rho ) / len(rho)


##################################################  HyperOpt functions  ##########################################################################
def hyperopt_main(x_var): 
    lr = x_var['x_lr']
    neuron = int(x_var['neuron'])
    CNN_dev= x_var['CNN_dev']
    rff_dev = x_var['rff_dev']
    iteration = int(x_var['No_iteration'])
    N_Layers = int(x_var['N_Layers'])
    act_func = x_var['act_func']
    
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom, boundary_neumann, boundary_dirichlet = get_Train_domain()
    x, y, datatest = get_Test_datatest( num_test_x , num_test_y )
    
    #--- Activate for circular inclusion-----
    #density= get_density()
    density=1
    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------
     
    dem = DeepEnergyMethod([D_in, neuron, D_out], 2, E, nu, act_func, CNN_dev,rff_dev,N_Layers)
    
    # ----------------------------------------------------------------------
    #                   STEP 3: TRAINING MODEL
    # ----------------------------------------------------------------------
    Loss= dem.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet, iteration, lr,N_Layers,act_func,density)
    print('lr: %.5e,\t neuron: %.3d, \t CNN_Sdev: %.5e, \t RNN_Sdev: %.5e, \t Itertions: %.3d, \t Layers: %d, \t Act_fn : %s,\t Loss: %.5e'
      % (lr, neuron, CNN_dev,rff_dev,iteration,N_Layers, act_func,Loss))

    f.write('lr: %.5e,\t neuron: %.3d, \t CNN_Sdev: %.5e, \t RNN_Sdev: %.5e, \t Itertions: %.3d, \t Layers: %d, \t Act_fn : %s,\t Loss: %.5e'
      % (lr, neuron, CNN_dev,rff_dev,iteration,N_Layers, act_func,Loss))

    # ----------------------------------------------------------------------
    #                   STEP 4: TEST MODEL
    # ----------------------------------------------------------------------
    
    U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises = dem.evaluate_model(x, y, E, nu, N_Layers,act_func)
    surUx, surUy, surUz = U    
    return Loss
    

#------------------------- Example number ----------------
global Example
# 'Shear' , 'Bulk' , 'Poisson'
Example = 'Shear'

#------------------------- Constant Network Parameters ----------------
D_in = 2
D_out = 2

# -------------------------- Structural Parameters ---------------------
Length = 10.
Height = 10. 
Depth = 1.0


# -------------------------- Material Parameters -----------------------
E = 2e5 # Pa
nu = 0.3


# ------------------------- Datapoints for training ---------------------
Nx = 80 + 1
Ny = 80 + 1


x_min, y_min = (0.0, 0.0)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
shape = [Nx, Ny]
dxdy = [hx, hy]

# ------------------------- Datapoints for evaluation -----------------
Length_test= Length
Height_test= Height
num_test_x = Nx
num_test_y = Ny
hx_test = Length / (num_test_x - 1)
hy_test = Height / (num_test_y - 1)
shape_test=[num_test_x,num_test_y]


# ------------------------- Perform hyper parameter optimization or not -----------------
HyperOPT = False

#######################################################################################
# Begin Hyper Opt
if HyperOPT:
    #-------------------------- File to write results in ---------
    if os.path.exists("HOpt_Runs.txt"):
      os.remove("HOpt_Runs.txt")

    f = open("HOpt_Runs.txt", "a")

    #-------------------------- Variable HyperParameters-----------------------------
    space = {
        'x_lr': hp.loguniform('x_lr', 0, 2),
        'neuron': 2*hp.quniform('neuron', 10, 60, 1),
        'act_func': hp.choice('act_func', ['tanh','relu','rrelu','sigmoid']),
        'CNN_dev': hp.uniform('CNN_dev', 0, 1),
        'rff_dev': hp.uniform('rff_dev', 0, 1),
        'No_iteration': hp.quniform('No_iteration', 40, 100, 1),
        'N_Layers':hp.quniform('N_Layers',3,5,1)
    }

    trials = Trials()
    best = fmin(hyperopt_main,
                space,
                algo=tpe.suggest,
                max_evals=100,
                trials=Trials(),
                rstate = np.random.default_rng(2019),
                max_queue_len=2
                )
    print(best)



#######################################################################################
# Begin TO:
# Optimal hyper parameters for baseline config
lr = 1.73553
neuron = 68
CNN_dev = 0.062264
rff_dev = 0.119297
iteration = 100
N_Layers = 5
act_func ='rrelu'


Vf = 0.5
max_TO_itr = 150
filename_out = './Example' + str(Example) + '/'
# convergence_tol = 5. * np.power( 10. , np.linspace( -1. , -6. , max_TO_itr ) )
convergence_tol = 5e-5 * np.ones(max_TO_itr)
verbose = True

# ----------------------------------------------------------------------
#                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
# ----------------------------------------------------------------------
dom, boundary_neumann, boundary_dirichlet = get_Train_domain()
x, y, datatest = get_Test_datatest( num_test_x , num_test_y )

# ----------------------------------------------------------------------
#                   STEP 2: SETUP MODEL
# ----------------------------------------------------------------------
 
dem = DeepEnergyMethod([D_in, neuron, D_out], 2, E, nu, act_func, CNN_dev,rff_dev,N_Layers)

# TO using MMA
start_t = time.time()
W = Filter( 0.25 )
end_t = time.time()
print( 'Generating density filter took ' + str( end_t - start_t ) + ' s' )

# Passive Density elements for topology opt.
density = np.ones( (Ny-1)*(Nx-1) ) * Vf


# Define initial density
x = np.linspace( 0. , Length , Nx-1 )
y = np.linspace( 0. , Height , Ny-1 )
X , Y = np.meshgrid( x , y )
X = X.T; Y = Y.T

density = density.reshape( [ Ny-1,Nx-1 ] )
mask = np.sqrt( ( X - Length/2. )**2 + ( Y - Height/2. )**2 ) <= Height / 5.
density[ mask ] = 0.
density = density.flatten()



start_t = time.time()
TO_itr = 0
Train_time , ObjVal = [] , []
optimizationParams = {'maxIters':max_TO_itr,'minIters':1,'relTol':0.01}
final_rho , final_obj = optimize( density , optimizationParams , Val_and_Grad , VolCon , Ny-1 , Nx-1 )
end_t = time.time()
t_tot = end_t - start_t
print( 'Topology optimization took ' + str( t_tot ) + ' s' )


plt.figure()
plt.plot( np.arange(len(Train_time)) + 1 , Train_time )
plt.xlabel('TopOpt iteration')
plt.ylabel('DEM train time [s]')
plt.title('Total TO time = ' + str(t_tot) + 's' )
plt.savefig( filename_out + 'Train_time.png' , dpi=600 )

plt.figure()
plt.plot( np.arange(len(ObjVal)) + 1 , ObjVal )
plt.xlabel('TopOpt iteration')
plt.ylabel('Compliance [J]')
plt.savefig( filename_out + 'Compliance.png' , dpi=600 )
np.save( filename_out + 'Compliance.npy' , ObjVal )
plt.close()

Train_time.append( t_tot )
np.save( filename_out + 'Train_time.npy' , Train_time )