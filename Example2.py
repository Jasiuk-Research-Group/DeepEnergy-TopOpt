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
    z_dom = z_min, Depth, Nz
    # create points
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
    lin_z = np.linspace(z_dom[0], z_dom[1], z_dom[2])
    dom = np.zeros((Nx * Ny * Nz, 3))
    c = 0
    for z in np.nditer(lin_z):
        for x in np.nditer(lin_x):
            tb = y_dom[2] * c
            te = tb + y_dom[2]
            c += 1
            dom[tb:te, 0] = x
            dom[tb:te, 1] = lin_y
            dom[tb:te, 2] = z
    print( 'Domain shape (nodes): ' + str( dom.shape ) )
    
    # ------------------------------------ BOUNDARY ----------------------------------------
    # Left boundary condition (Dirichlet BC)
    bcl_u_pts_idx = np.where(dom[:, 0] == x_min) # Index/ node numbers at which x=x_min
    bcl_u_pts = dom[bcl_u_pts_idx, :][0] #Coordinates at which x=xmin
    bcl_u = np.ones(np.shape(bcl_u_pts)) * [0., 0., 0.] #Define displacement constraints at the nodes


    if Example == 1:
        # Right boundary condition (Dirichlet BC)
        bcr_u_pts_idx = np.where(dom[:, 0] == Length)
        bcr_u_pts = dom[bcr_u_pts_idx, :][0]
        bcr_u = np.ones(np.shape(bcr_u_pts)) * [0., 0., 0.]

        # Downward load from the middle of the domain
        len_load = 0.5
        bcr_t_pts_idx = np.where( (dom[:, 0] >= Length/2. - len_load/2.) & (dom[:, 0] <= Length/2. + len_load/2.) & (dom[:, 1] == Height ) & (dom[:, 2] >= Depth/2. - len_load/2.) & (dom[:, 2] <= Depth/2. + len_load/2.) )
        bcr_t_pts = dom[bcr_t_pts_idx, :][0]
        bcr_t = np.ones(np.shape(bcr_t_pts)) * [0., -2000., 0.]

        boundary_neumann = {
            # condition on the right
            "neumann_1": {
                "coord": bcr_t_pts,
                "known_value": bcr_t,
                "penalty": 1.,
                "idx":np.asarray(bcr_t_pts_idx)
            }
            # adding more boundary condition here ...
        }
        
        boundary_dirichlet = {
            # condition on the left
            "dirichlet_1": {
                "coord": bcl_u_pts,
                "known_value": bcl_u,
                "penalty": 1.,
                "idx":np.asarray(bcl_u_pts_idx)
            } ,
            "dirichlet_2": {
                "coord": bcr_u_pts,
                "known_value": bcr_u,
                "penalty": 1.,
                "idx":np.asarray(bcr_u_pts_idx)
            } ,
            # adding more boundary condition here ...
        }
    elif Example == 2:
        # Right boundary condition (Neumann BC)
        bcr_t_pts_idx = np.where( (dom[:, 0] == Length) & ( dom[:, 1] <= Height / 4. ) )
        bcr_t_pts = dom[bcr_t_pts_idx, :][0]
        bcr_t = np.ones(np.shape(bcr_t_pts)) * [0., -200., 0.]

        boundary_neumann = {
            # condition on the right
            "neumann_1": {
                "coord": bcr_t_pts,
                "known_value": bcr_t,
                "penalty": 1.,
                "idx":np.asarray(bcr_t_pts_idx)
            }
            # adding more boundary condition here ...
        }
        
        boundary_dirichlet = {
            # condition on the left
            "dirichlet_1": {
                "coord": bcl_u_pts,
                "known_value": bcl_u,
                "penalty": 1.,
                "idx":np.asarray(bcl_u_pts_idx)
            }
        }

    return dom, boundary_neumann, boundary_dirichlet

def get_Test_datatest(Nx, Ny, Nz):
    x_dom_test = x_min, Length_test, Nx
    y_dom_test = y_min, Height_test, Ny
    z_dom_test = z_min, Depth_test, Nz
    # create points
    x_space = np.linspace(x_dom_test[0], x_dom_test[1], x_dom_test[2])
    y_space = np.linspace(y_dom_test[0], y_dom_test[1], y_dom_test[2])
    z_space = np.linspace(z_dom_test[0], z_dom_test[1], z_dom_test[2])
    xGrid, yGrid, zGrid = np.meshgrid(x_space, y_space, z_space)
    data_test = np.concatenate(
        (np.array([xGrid.flatten()]).T, np.array([yGrid.flatten()]).T, np.array([zGrid.flatten()]).T), axis=1)
    return x_space, y_space, z_space, data_test

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
        # self.data = data
        self.model = MultiLayerNet(model[0], model[1], model[2],act_func, CNN_dev, rff_dev,N_Layers)
        self.model = self.model.to(dev)
        self.InternalEnergy= InternalEnergy(E, nu)
        self.FextLoss = IntegrationFext(dim)
        self.dim = dim
        self.lossArray = []

    def train_model(self, shape, dxdydz, data, neumannBC, dirichletBC, iteration, learning_rate,N_Layers,activatn_fn, rho , TO_itr ):            
        global Train_time , ObjVal
        x = torch.from_numpy(data).float()
        x = x.to(dev)
        x.requires_grad_(True)

        density = torch.from_numpy(rho).float()
        density = torch.reshape( density , [ Nz-1 , Ny-1 , Nx-1 ] ).to(dev)


        # get tensor inputs and outputs for boundary conditions
        # -------------------------------------------------------------------------------
        #                           Neumann BC
        # -------------------------------------------------------------------------------
        neuBC_coordinates = {}  # declare a dictionary
        neuBC_values = {}  # declare a dictionary
        neuBC_penalty = {}
        neuBC_Zeros= {}
        neuBC_idx={}
        
        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).float().to(dev)
            neuBC_coordinates[i].requires_grad_(True)
            neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).float().to(dev)
            neuBC_penalty[i] = torch.tensor(neumannBC[keyi]['penalty']).float().to(dev)
            neuBC_idx[i]=torch.from_numpy(neumannBC[keyi]['idx']).float().to(dev)


        # ----------------------------------------------------------------------------------
        # Minimizing loss function (energy and boundary conditions)
        # ----------------------------------------------------------------------------------
        optimizer_LBFGS = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=20,line_search_fn='strong_wolfe')
        optimizer_Adam = torch.optim.Adam(self.model.parameters(), lr=0.001)
        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        loss_history= np.zeros(iteration)
        Iter_No_Hist= []

        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                it_time = time.time()
                u_pred = self.getU(x,N_Layers,activatn_fn)
                u_pred.double()

                # ---- Calculate internal and external energies------                 
                storedEnergy= self.InternalEnergy.Elastic3DGaussQuad(u_pred, x, dxdydz, shape, density , 0 )
                externalE = self.FextLoss.lossFextEnergy(u_pred, x, neuBC_coordinates, neuBC_values, neuBC_idx, dxdydz)
                energy_loss = storedEnergy - externalE

                loss = energy_loss
                optimizer_LBFGS.zero_grad()
                loss.backward()

                if verbose:
                    print('     Iter: %d Loss: %.6e IntE: %.4e ExtE: %.4e'% (t + 1, loss.item(), storedEnergy.item(),externalE.item() ))
                loss_history[t]= energy_loss.data
                energy_loss_array.append(energy_loss.data)
                Iter_No_Hist.append(t)
                self.lossArray.append( loss.data.cpu() )

                if np.isnan( self.lossArray[-1] ):
                    print('WARNING: NaN detected in loss!' )
                return loss
            optimizer_LBFGS.step(closure)

            # Check convergence
            if ConvergenceCheck( self.lossArray , convergence_tol[ TO_itr ] ):
                break
        elapsed = time.time() - start_time
        # plt.figure()
        # plt.scatter(np.linspace(1,iteration,iteration), loss_history)
        # plt.draw()
        # plt.pause( 3. )
        # plt.close()
        print('Training time: %.4f' % elapsed)
        Train_time.append( elapsed )

        # TO:
        u_pred = self.getU(x,N_Layers,activatn_fn) #                                                     Sensitivity mode
        dfdrho , compliance = self.InternalEnergy.Elastic3DGaussQuad(u_pred, x, dxdydz, shape, density , 1 )
        ObjVal.append( compliance.cpu().detach().numpy() )
        return dfdrho , compliance


    def getU(self, x,N_Layers,activatn_fn):
        u = self.model( x , N_Layers,activatn_fn)
        phix = x[:, 0] / Length

        if Example == 1:
            Ux = phix * ( 1. - phix ) * u[:, 0]
            Uy = phix * ( 1. - phix ) * u[:, 1]
            Uz = phix * ( 1. - phix ) * u[:, 2]
        elif Example == 2:
            Ux = phix * u[:, 0]
            Uy = phix * u[:, 1]
            Uz = phix * u[:, 2]

        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        Uz = Uz.reshape(Uz.shape[0], 1)
        u_pred = torch.cat((Ux, Uy, Uz), -1)
        return u_pred
    
    def evaluate_model(self, data , dev , dxdydz , shape , N_Layers,activatn_fn , rho , fn ):  
        density = torch.from_numpy(rho).float()
        density = torch.reshape( density , [ Nz-1 , Ny-1 , Nx-1 ] ).to(dev)

        nodes = torch.from_numpy(data).float()
        nodes = nodes.to(dev)

        # Get displacement
        u_pred_torch = self.getU( nodes ,N_Layers,activatn_fn)
        # Stress and strain
        curr_Strain , curr_Stress = self.InternalEnergy.Elastic3DGaussQuad(u_pred_torch, nodes, dxdydz, shape, density , -1 )
        curr_Strain = curr_Strain.detach().cpu().numpy()
        curr_Stress = curr_Stress.detach().cpu().numpy()
        density = density.detach().cpu().numpy()

        # Unpack as numpy arrays
        order = [ shape[-1] , shape[0] , shape[1] ]
        Ux = torch.transpose(u_pred_torch[:, 0].reshape( order ), 1, 2).detach().cpu().numpy()
        Uy = torch.transpose(u_pred_torch[:, 1].reshape( order ), 1, 2).detach().cpu().numpy()
        Uz = torch.transpose(u_pred_torch[:, 2].reshape( order ), 1, 2).detach().cpu().numpy()

        Px = torch.transpose(nodes[:, 0].reshape( order ), 1, 2).detach().cpu().numpy()
        Py = torch.transpose(nodes[:, 1].reshape( order ), 1, 2).detach().cpu().numpy()
        Pz = torch.transpose(nodes[:, 2].reshape( order ), 1, 2).detach().cpu().numpy()

        E11 = curr_Strain[0]
        E22 = curr_Strain[1]
        E33 = curr_Strain[2]
        E23 = curr_Strain[3]
        E13 = curr_Strain[4]
        E12 = curr_Strain[5]

        S11 = curr_Stress[0]
        S22 = curr_Stress[1]
        S33 = curr_Stress[2]
        S23 = curr_Stress[3]
        S13 = curr_Stress[4]
        S12 = curr_Stress[5]

        SVonMises = np.float64( np.sqrt(0.5 * ((S11 - S22) ** 2 + (S22) ** 2 + (-S11) ** 2 + 6 * (S12 ** 2))) )

        util.write_vtk_v2( fn, Px, Py, Pz, Ux,Uy,Uz, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises , density )

 


##################################################  TopOpt functions  ##########################################################################
def Filter( nodes , shape , rad ):
    nex = Nx-1; ney = Ny-1; nez = Nz-1
    Lx = Length
    Ly = Height
    Lz = Depth
    dx = Lx / nex
    dy = Ly / ney
    dz = Lz / nez

    xx = np.linspace(0,Lx,nex)
    yy = np.linspace(0,Ly,ney)
    zz = np.linspace(0,Lz,nez)
    Y , Z , X = np.meshgrid( yy , zz , xx )
    X = X.flatten(); Y = Y.flatten(); Z = Z.flatten()

    wi , wj , wv = [] , [] , []
    for eid in range( nex * ney * nez ):
        my_X = X[eid]; my_Y = Y[eid]; my_Z = Z[eid]
        dist = np.sqrt( ( X - my_X )**2 + ( Y - my_Y )**2 + ( Z - my_Z )**2 )
        neighbours = np.where( dist <= rad )[0]
        wi += [eid] * len(neighbours)
        wj += list( neighbours )
        wv += list( rad - dist[ neighbours ] )
    
    W = normalize( coo_matrix( (wv, (wi, wj)), shape=(nex*ney*nez, nex*ney*nez)) , norm='l1', axis=1).tocsr() # Normalize row-wise
    return W


def Val_and_Grad( density ):
    global TO_itr
    dxdydz = [hx, hy, hz]
    rho_tilda = W @ density 
    # rho_tilda = density 

    dfdrho_tilda , compliance = dem.train_model(shape, dxdydz, dom, boundary_neumann, boundary_dirichlet, iteration, lr,N_Layers,act_func,rho_tilda , TO_itr )
    dfdrho_tilda_npy = dfdrho_tilda.cpu().detach().numpy()

    # Invert filter
    ss = dfdrho_tilda_npy.shape
    sensitivity = W.T @ dfdrho_tilda_npy.flatten()
    # sensitivity = dfdrho_tilda_npy.flatten()


    # Save plot
    fig, axs = plt.subplots(2, 1, sharex='col')
    ff = axs[0].imshow( np.flipud( np.mean( density.reshape(ss) , axis=0 ) ) , extent = [x_min, Length, y_min, Height],cmap='binary')
    plt.colorbar( ff , ax=axs[0] )
    axs[0].set_aspect('equal') 
    axs[0].set_title('Element density')
    ff = axs[1].imshow( np.flipud( np.mean( sensitivity.reshape(ss) , axis=0 ) ) , extent = [x_min, Length, y_min, Height],cmap='jet')
    plt.colorbar( ff , ax=axs[1] )
    axs[1].set_aspect('equal') 
    axs[1].set_title('Sensitivity' )
    plt.savefig( './Example' + str(Example) + '/Design_' + str(TO_itr) + '.png' , format = 'png', dpi=600 , bbox_inches="tight" )
    plt.close()
    np.save('./Example' + str(Example) + '/Density_' + str(TO_itr) + '.npy' , density.reshape(ss) )

    
    # Save some data
    fn = filename_out + 'TO_itr_' + str(TO_itr)
    dem.evaluate_model( dom,dev,dxdydz,shape,N_Layers,act_func, rho_tilda , fn )

    # exit()

    TO_itr += 1
    return compliance.cpu().detach().numpy() , sensitivity


# Volume constraint
def VolCon( rho ):
    return np.mean( rho ) - 0.4 , np.ones_like( rho ) / len(rho)


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
Example = 1

#------------------------- Constant Network Parameters ----------------
D_in = 3
D_out = 3

# -------------------------- Structural Parameters ---------------------
# Example = 1
Length = 12.
Height = 2.
if Example == 2:
    Length = 10.
    Height = 5.   
Depth = 2.0


# -------------------------- Material Parameters -----------------------
E = 2e5 # Pa
nu = 0.3


# ------------------------- Datapoints for training ---------------------
# Example = 1
Nx = 120 + 1
Ny = 25 + 1
Nz = 25 + 1
if Example == 2:
    Nx = 100 + 1
    Ny = 50 + 1
    Nz = 2 + 1


x_min, y_min, z_min = (0.0, 0.0, 0.0)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
hz = Depth / (Nz - 1)
shape = [Nx, Ny, Nz]
dxdyz = [hx, hy, hz]

# ------------------------- Datapoints for evaluation -----------------
Length_test= Length
Height_test= Height
Depth_test= Depth
num_test_x = Nx
num_test_y = Ny
num_test_z = Nz
hx_test = Length / (num_test_x - 1)
hy_test = Height / (num_test_y - 1)
hz_test = Depth / (num_test_z - 1)
shape_test=[num_test_x,num_test_y,num_test_z]


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


max_TO_itr = 80
filename_out = './Example' + str(Example) + '/'
# convergence_tol = 5. * np.power( 10. , np.linspace( -1. , -6. , max_TO_itr ) )
convergence_tol = 5e-6 * np.ones(max_TO_itr)
verbose = False

# ----------------------------------------------------------------------
#                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
# ----------------------------------------------------------------------
dom, boundary_neumann, boundary_dirichlet = get_Train_domain()
x, y, z , datatest = get_Test_datatest( num_test_x , num_test_y , num_test_z )

# ----------------------------------------------------------------------
#                   STEP 2: SETUP MODEL
# ----------------------------------------------------------------------
 
dem = DeepEnergyMethod([D_in, neuron, D_out], 3 , E, nu, act_func, CNN_dev,rff_dev,N_Layers)

# TO using MMA
start_t = time.time()
W = Filter( datatest , shape , 0.3 )
end_t = time.time()
print( 'Generating density filter took ' + str( end_t - start_t ) + ' s' )

# Passive Density elements for topology opt.
density = np.ones( (Ny-1)*(Nx-1)*(Nz-1) ) * 0.4


start_t = time.time()
TO_itr = 0
Train_time , ObjVal = [] , []
optimizationParams = {'maxIters':max_TO_itr,'minIters':2,'relTol':0.01}
final_rho , final_obj = optimize( density , optimizationParams , Val_and_Grad , VolCon , Ny-1 , Nx-1 , Nz-1 )
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
plt.plot( np.arange(len(Train_time)) + 1 , ObjVal )
plt.xlabel('TopOpt iteration')
plt.ylabel('Compliance [J]')
plt.savefig( filename_out + 'Compliance.png' , dpi=600 )
np.save( filename_out + 'Compliance.npy' , ObjVal )
plt.close()

Train_time.append( t_tot )
np.save( filename_out + 'Train_time.npy' , Train_time )
