from Sub_Functions.config import *
import numpy as np


class EnergyModel:
    # ---------------------------------------------------------------------------------------------------------------------------
    
    ##-- Propose Change: Add another def for linear model ---##
    
    def __init__(self, energy, dim, E=None, nu=None, param_c1=None, param_c2=None, param_c=None):
        self.type = energy
        self.dim = dim
        if self.type == 'neohookean':
            self.mu = E / (2 * (1 + nu))
            self.lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        if self.type == 'mooneyrivlin':
            self.param_c1 = param_c1
            self.param_c2 = param_c2
            self.param_c = param_c
            self.param_d = 2 * (self.param_c1 + 2 * self.param_c2)
        # Modify Charul
        if self.type=='elastic':
            self.E= E
            self.nu=nu

    def getStoredEnergy(self, u, x):
    #def getStoredEnergy(self, u, x, dxdydz, shape):
        if self.type == 'neohookean':
            if self.dim == 2:
                return self.NeoHookean2D(u, x)
            if self.dim == 3:
                return self.NeoHookean3D(u, x)
        if self.type == 'mooneyrivlin':
            if self.dim == 2:
                return self.MooneyRivlin2D(u, x)
            if self.dim == 3:
                return self.MooneyRivlin3D(u, x)
        #Modify Charul
        if self.type == 'elastic':
            if self.dim == 2:
                return self.Elastic2D(u, x)
                #return self.Elastic2DGauusQuad(self, u, x, dxdydz, shape)
            if self.dim == 3:
                return self.Elastic3D(u, x)
    
    def getGaussStoredEnergy(self, u, x, dxdydz, shape,density):
        if self.type == 'elastic':
            if self.dim == 2:
                return self.Elastic2DGauusQuad(u, x, dxdydz, shape,density)
    
    def Elastic2D (self, u, x):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        
        e_xx= duxdxy[:, 0].unsqueeze(1)
        e_yy= duydxy[:, 1].unsqueeze(1)
        e_xy= (duxdxy[:, 1].unsqueeze(1)+ duydxy[:, 0].unsqueeze(1))/2
                
        stress_xx= self.E*(e_xx+ self.nu*e_yy)/(1-self.nu**2)
        stress_yy= self.E*(e_yy+ self.nu*e_xx)/(1-self.nu**2)
        stress_xy= self.E*e_xy/(1+self.nu)
        
        strainEnergy= 0.5*(e_xx*stress_xx+ e_yy*stress_yy+ 2*e_xy*stress_xy)
               
        return strainEnergy
    
    def Elastic2DGauusQuad (self, u, x, dxdydz, shape,density):
              
        Ux= torch.transpose(u[:, 0].unsqueeze(1).reshape(shape[0], shape[1]), 0, 1)
        Uy= torch.transpose(u[:, 1].unsqueeze(1).reshape(shape[0], shape[1]), 0, 1)
        
        axis=-1
        
        nd = Ux.ndimension()
        slice1 = [slice(None)] * nd
        slice2 = [slice(None)] * nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        
        UxN1= Ux[:(shape[1]-1)][tuple(slice2)]
        UxN2= Ux[1:shape[1]][tuple(slice2)]
        UxN3= Ux[0:(shape[1]-1)][tuple(slice1)]
        UxN4= Ux[1:shape[1]][tuple(slice1)]
        
        UyN1= Uy[:(shape[1]-1)][tuple(slice2)]
        UyN2= Uy[1:shape[1]][tuple(slice2)]
        UyN3= Uy[0:(shape[1]-1)][tuple(slice1)]
        UyN4= Uy[1:shape[1]][tuple(slice1)]
        
        ## Differentiation of shape functions at gauss quadrature points       
        
        dN1_dsy=np.array([[-0.394337567,-0.105662433,-0.105662433,-0.394337567],[-0.394337567,-0.394337567,-0.105662433,-0.105662433]])
        dN2_dsy=np.array([[-0.105662433,-0.394337567,-0.394337567,-0.105662433],[0.394337567,0.394337567,0.105662433,0.105662433]])
        dN3_dsy=np.array([[0.394337567,0.105662433,0.105662433,0.394337567],[-0.105662433,-0.105662433,-0.394337567,-0.394337567]])
        dN4_dsy=np.array([[0.105662433,0.394337567,0.394337567,0.105662433],[0.105662433,0.105662433,0.394337567,0.394337567]])        
       
        dx=dxdydz[0]
        dy=dxdydz[1]
        dxds=dx/2
        dydt=dy/2
        
        J= np.array([[dxds,0],[0,dydt]])
        Jinv= np.linalg.inv(J)
        detJ= np.linalg.det(J)
        
        GaussPoints=4
        #density= torch.ones(UyN1.shape)
    
        #strainEnergy_GP=torch.zeros((GaussPoints,1))
        
        
        #for gp in range(GaussPoints):
			
            # dN1_dxy_Gp=np.matmul(Jinv, np.array([dN1_dsy[0][gp],dN1_dsy[1][gp]]).reshape((2,1)))
            # dN2_dxy_Gp=np.matmul(Jinv, np.array([dN2_dsy[0][gp],dN2_dsy[1][gp]]).reshape((2,1)))
            # dN3_dxy_Gp=np.matmul(Jinv, np.array([dN3_dsy[0][gp],dN3_dsy[1][gp]]).reshape((2,1)))
            # dN4_dxy_Gp=np.matmul(Jinv, np.array([dN4_dsy[0][gp],dN4_dsy[1][gp]]).reshape((2,1)))
            
            # dUxdx_GP= dN1_dxy_Gp[0][0]*UxN1+ dN2_dxy_Gp[0][0]*UxN2+ dN3_dxy_Gp[0][0]*UxN3+ dN4_dxy_Gp[0][0]*UxN4
            # dUxdy_GP= dN1_dxy_Gp[1][0]*UxN1+ dN2_dxy_Gp[1][0]*UxN2+ dN3_dxy_Gp[1][0]*UxN3+ dN4_dxy_Gp[1][0]*UxN4
            
            # dUydx_GP= dN1_dxy_Gp[0][0]*UyN1+ dN2_dxy_Gp[0][0]*UyN2+ dN3_dxy_Gp[0][0]*UyN3+ dN4_dxy_Gp[0][0]*UyN4
            # dUydy_GP= dN1_dxy_Gp[1][0]*UyN1+ dN2_dxy_Gp[1][0]*UyN2+ dN3_dxy_Gp[1][0]*UyN3+ dN4_dxy_Gp[1][0]*UyN4
                    
        
            # #Strains at all gauss quadrature points
            # e_xx_GP= dUxdx_GP
            # e_yy_GP= dUydy_GP
            # e_xy_GP= 0.5*(dUydx_GP+dUxdy_GP)
                    
            # #Stresses at all gauss quadrature points
        
            # S_xx_GP= self.E*(e_xx_GP+ self.nu*e_yy_GP)/(1-self.nu**2)        
            # S_yy_GP= self.E*(e_yy_GP+ self.nu*e_xx_GP)/(1-self.nu**2)
            # S_xy_GP= self.E*e_xy_GP/(1+self.nu)
            
        
            # strainEnergy_GP[gp][0]= torch.sum(0.5*(e_xx_GP*S_xx_GP+ e_yy_GP*S_yy_GP+ 2*e_xy_GP*S_xy_GP))*detJ
    
        #strainEnergy= sum(strainEnergy_GP)
        
        
        ## Strain energy at GP1
        dN1_dxy_Gp1=np.matmul(Jinv, np.array([dN1_dsy[0][0],dN1_dsy[1][0]]).reshape((2,1)))
        dN2_dxy_Gp1=np.matmul(Jinv, np.array([dN2_dsy[0][0],dN2_dsy[1][0]]).reshape((2,1)))
        dN3_dxy_Gp1=np.matmul(Jinv, np.array([dN3_dsy[0][0],dN3_dsy[1][0]]).reshape((2,1)))
        dN4_dxy_Gp1=np.matmul(Jinv, np.array([dN4_dsy[0][0],dN4_dsy[1][0]]).reshape((2,1)))
           
        dUxdx_GP1= dN1_dxy_Gp1[0][0]*UxN1+ dN2_dxy_Gp1[0][0]*UxN2+ dN3_dxy_Gp1[0][0]*UxN3+ dN4_dxy_Gp1[0][0]*UxN4
        dUxdy_GP1= dN1_dxy_Gp1[1][0]*UxN1+ dN2_dxy_Gp1[1][0]*UxN2+ dN3_dxy_Gp1[1][0]*UxN3+ dN4_dxy_Gp1[1][0]*UxN4
           
        dUydx_GP1= dN1_dxy_Gp1[0][0]*UyN1+ dN2_dxy_Gp1[0][0]*UyN2+ dN3_dxy_Gp1[0][0]*UyN3+ dN4_dxy_Gp1[0][0]*UyN4
        dUydy_GP1= dN1_dxy_Gp1[1][0]*UyN1+ dN2_dxy_Gp1[1][0]*UyN2+ dN3_dxy_Gp1[1][0]*UyN3+ dN4_dxy_Gp1[1][0]*UyN4
                   
        
        #Strains at all gauss quadrature points
        e_xx_GP1= dUxdx_GP1
        e_yy_GP1= dUydy_GP1
        e_xy_GP1= 0.5*(dUydx_GP1+dUxdy_GP1)
                
        #Stresses at all gauss quadrature points
        
        S_xx_GP1= (self.E*(e_xx_GP1+ self.nu*e_yy_GP1)/(1-self.nu**2))*density        
        S_yy_GP1= (self.E*(e_yy_GP1+ self.nu*e_xx_GP1)/(1-self.nu**2))*density
        S_xy_GP1= (self.E*e_xy_GP1/(1+self.nu))*density
        
        strainEnergy_GP1= torch.sum(0.5*(e_xx_GP1*S_xx_GP1+ e_yy_GP1*S_yy_GP1+ 2*e_xy_GP1*S_xy_GP1))*detJ
        
        ## Strain energy at GP2
        dN1_dxy_Gp2=np.matmul(Jinv, np.array([dN1_dsy[0][1],dN1_dsy[1][1]]).reshape((2,1)))
        dN2_dxy_Gp2=np.matmul(Jinv, np.array([dN2_dsy[0][1],dN2_dsy[1][1]]).reshape((2,1)))
        dN3_dxy_Gp2=np.matmul(Jinv, np.array([dN3_dsy[0][1],dN3_dsy[1][1]]).reshape((2,1)))
        dN4_dxy_Gp2=np.matmul(Jinv, np.array([dN4_dsy[0][1],dN4_dsy[1][1]]).reshape((2,1)))
        
        dUxdx_GP2= dN1_dxy_Gp2[0][0]*UxN1+ dN2_dxy_Gp2[0][0]*UxN2+ dN3_dxy_Gp2[0][0]*UxN3+ dN4_dxy_Gp2[0][0]*UxN4
        dUxdy_GP2= dN1_dxy_Gp2[1][0]*UxN1+ dN2_dxy_Gp2[1][0]*UxN2+ dN3_dxy_Gp2[1][0]*UxN3+ dN4_dxy_Gp2[1][0]*UxN4
        
        dUydx_GP2= dN1_dxy_Gp2[0][0]*UyN1+ dN2_dxy_Gp2[0][0]*UyN2+ dN3_dxy_Gp2[0][0]*UyN3+ dN4_dxy_Gp2[0][0]*UyN4
        dUydy_GP2= dN1_dxy_Gp2[1][0]*UyN1+ dN2_dxy_Gp2[1][0]*UyN2+ dN3_dxy_Gp2[1][0]*UyN3+ dN4_dxy_Gp2[1][0]*UyN4
                
        
        #Strains at all gauss quadrature points
        e_xx_GP2= dUxdx_GP2
        e_yy_GP2= dUydy_GP2
        e_xy_GP2= 0.5*(dUydx_GP2+dUxdy_GP2)
                
        #Stresses at all gauss quadrature points
        
        S_xx_GP2= (self.E*(e_xx_GP2+ self.nu*e_yy_GP2)/(1-self.nu**2))*density        
        S_yy_GP2= (self.E*(e_yy_GP2+ self.nu*e_xx_GP2)/(1-self.nu**2))*density
        S_xy_GP2= (self.E*e_xy_GP2/(1+self.nu))*density
        
        strainEnergy_GP2= torch.sum(0.5*(e_xx_GP2*S_xx_GP2+ e_yy_GP2*S_yy_GP2+ 2*e_xy_GP2*S_xy_GP2))*detJ
        
        ## Strain energy at GP3
        dN1_dxy_GP3=np.matmul(Jinv, np.array([dN1_dsy[0][2],dN1_dsy[1][2]]).reshape((2,1)))
        dN2_dxy_GP3=np.matmul(Jinv, np.array([dN2_dsy[0][2],dN2_dsy[1][2]]).reshape((2,1)))
        dN3_dxy_GP3=np.matmul(Jinv, np.array([dN3_dsy[0][2],dN3_dsy[1][2]]).reshape((2,1)))
        dN4_dxy_GP3=np.matmul(Jinv, np.array([dN4_dsy[0][2],dN4_dsy[1][2]]).reshape((2,1)))
        
        dUxdx_GP3= dN1_dxy_GP3[0][0]*UxN1+ dN2_dxy_GP3[0][0]*UxN2+ dN3_dxy_GP3[0][0]*UxN3+ dN4_dxy_GP3[0][0]*UxN4
        dUxdy_GP3= dN1_dxy_GP3[1][0]*UxN1+ dN2_dxy_GP3[1][0]*UxN2+ dN3_dxy_GP3[1][0]*UxN3+ dN4_dxy_GP3[1][0]*UxN4
        
        dUydx_GP3= dN1_dxy_GP3[0][0]*UyN1+ dN2_dxy_GP3[0][0]*UyN2+ dN3_dxy_GP3[0][0]*UyN3+ dN4_dxy_GP3[0][0]*UyN4
        dUydy_GP3= dN1_dxy_GP3[1][0]*UyN1+ dN2_dxy_GP3[1][0]*UyN2+ dN3_dxy_GP3[1][0]*UyN3+ dN4_dxy_GP3[1][0]*UyN4
                
        
        #Strains at all gauss quadrature points
        e_xx_GP3= dUxdx_GP3
        e_yy_GP3= dUydy_GP3
        e_xy_GP3= 0.5*(dUydx_GP3+dUxdy_GP3)
                
        #Stresses at all gauss quadrature points
        
        S_xx_GP3= (self.E*(e_xx_GP3+ self.nu*e_yy_GP3)/(1-self.nu**2))*density        
        S_yy_GP3= (self.E*(e_yy_GP3+ self.nu*e_xx_GP3)/(1-self.nu**2))*density
        S_xy_GP3= (self.E*e_xy_GP3/(1+self.nu))*density
        
        strainEnergy_GP3= torch.sum(0.5*(e_xx_GP3*S_xx_GP3+ e_yy_GP3*S_yy_GP3+ 2*e_xy_GP3*S_xy_GP3))*detJ
        
        ## Strain energy at GP4
        dN1_dxy_GP4=np.matmul(Jinv, np.array([dN1_dsy[0][3],dN1_dsy[1][3]]).reshape((2,1)))
        dN2_dxy_GP4=np.matmul(Jinv, np.array([dN2_dsy[0][3],dN2_dsy[1][3]]).reshape((2,1)))
        dN3_dxy_GP4=np.matmul(Jinv, np.array([dN3_dsy[0][3],dN3_dsy[1][3]]).reshape((2,1)))
        dN4_dxy_GP4=np.matmul(Jinv, np.array([dN4_dsy[0][3],dN4_dsy[1][3]]).reshape((2,1)))
        
        dUxdx_GP4= dN1_dxy_GP4[0][0]*UxN1+ dN2_dxy_GP4[0][0]*UxN2+ dN3_dxy_GP4[0][0]*UxN3+ dN4_dxy_GP4[0][0]*UxN4
        dUxdy_GP4= dN1_dxy_GP4[1][0]*UxN1+ dN2_dxy_GP4[1][0]*UxN2+ dN3_dxy_GP4[1][0]*UxN3+ dN4_dxy_GP4[1][0]*UxN4
        
        dUydx_GP4= dN1_dxy_GP4[0][0]*UyN1+ dN2_dxy_GP4[0][0]*UyN2+ dN3_dxy_GP4[0][0]*UyN3+ dN4_dxy_GP4[0][0]*UyN4
        dUydy_GP4= dN1_dxy_GP4[1][0]*UyN1+ dN2_dxy_GP4[1][0]*UyN2+ dN3_dxy_GP4[1][0]*UyN3+ dN4_dxy_GP4[1][0]*UyN4
                
        
        #Strains at all gauss quadrature points
        e_xx_GP4= dUxdx_GP4
        e_yy_GP4= dUydy_GP4
        e_xy_GP4= 0.5*(dUydx_GP4+dUxdy_GP4)
                
        #Stresses at all gauss quadrature points
        
        S_xx_GP4= (self.E*(e_xx_GP4+ self.nu*e_yy_GP4)/(1-self.nu**2))*density        
        S_yy_GP4= (self.E*(e_yy_GP4+ self.nu*e_xx_GP4)/(1-self.nu**2))*density
        S_xy_GP4= (self.E*e_xy_GP4/(1+self.nu))*density
        
        strainEnergy_GP4= torch.sum(0.5*(e_xx_GP4*S_xx_GP4+ e_yy_GP4*S_yy_GP4+ 2*e_xy_GP4*S_xy_GP4))*detJ
        
        
        
        strainEnergy= strainEnergy_GP1 +strainEnergy_GP2 +strainEnergy_GP3 +strainEnergy_GP4
        
        return strainEnergy

    def MooneyRivlin3D(self, u, x):
        duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
        Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
        Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
        Fyx = duydxyz[:, 0].unsqueeze(1) + 0
        Fyy = duydxyz[:, 1].unsqueeze(1) + 1
        Fyz = duydxyz[:, 2].unsqueeze(1) + 0
        Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
        Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
        Fzz = duzdxyz[:, 2].unsqueeze(1) + 1
        detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
        C11 = Fxx ** 2 + Fyx ** 2 + Fzx ** 2
        C12 = Fxx * Fxy + Fyx * Fyy + Fzx * Fzy
        C13 = Fxx * Fxz + Fyx * Fyz + Fzx * Fzz
        C21 = Fxy * Fxx + Fyy * Fyx + Fzy * Fzx
        C22 = Fxy ** 2 + Fyy ** 2 + Fzy ** 2
        C23 = Fxy * Fxz + Fyy * Fyz + Fzy * Fzz
        C31 = Fxz * Fxx + Fyz * Fyx + Fzz * Fzx
        C32 = Fxz * Fxy + Fyz * Fyy + Fzz * Fzy
        C33 = Fxz ** 2 + Fyz ** 2 + Fzz ** 2
        trC = C11 + C22 + C33
        trC2 = C11*C11 + C12*C21 + C13*C31 + C21*C12 + C22*C22 + C23*C32 + C31*C13 + C32*C23 + C33*C33
        I1 = trC
        I2 = 0.5 * (trC*trC - trC2)
        J = detF
        strainEnergy = self.param_c * (J - 1) ** 2 - self.param_d * torch.log(J) + self.param_c1 * (
                I1 - 3) + self.param_c2 * (I2 - 3)
        return strainEnergy


    def MooneyRivlin2D(self, u, x):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxy[:, 0].unsqueeze(1) + 1
        Fxy = duxdxy[:, 1].unsqueeze(1) + 0
        Fyx = duydxy[:, 0].unsqueeze(1) + 0
        Fyy = duydxy[:, 1].unsqueeze(1) + 1
        detF = Fxx * Fyy - Fxy * Fyx
        C11 = Fxx * Fxx + Fyx * Fyx
        C12 = Fxx * Fxy + Fyx * Fyy
        C21 = Fxy * Fxx + Fyy * Fyx
        C22 = Fxy * Fxy + Fyy * Fyy
        J = detF
        traceC = C11 + C22
        I1 = traceC
        trace_C2 = C11 * C11 + C12 * C21 + C21 * C12 + C22 * C22
        I2 = 0.5 * (traceC ** 2 - trace_C2)
        strainEnergy = self.param_c * (J - 1) ** 2 - self.param_d * torch.log(J) + self.param_c1 * (I1 - 2) + self.param_c2 * (I2 - 1)
        return strainEnergy

    # ---------------------------------------------------------------------------------------
    # Purpose: calculate Neo-Hookean potential energy in 3D
    # ---------------------------------------------------------------------------------------
    def NeoHookean3D(self, u, x):
        duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
        Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
        Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
        Fyx = duydxyz[:, 0].unsqueeze(1) + 0
        Fyy = duydxyz[:, 1].unsqueeze(1) + 1
        Fyz = duydxyz[:, 2].unsqueeze(1) + 0
        Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
        Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
        Fzz = duzdxyz[:, 2].unsqueeze(1) + 1
        detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
        trC = Fxx ** 2 + Fxy ** 2 + Fxz ** 2 + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 + Fzx ** 2 + Fzy ** 2 + Fzz ** 2
        strainEnergy = 0.5 * self.lam * (torch.log(detF) * torch.log(detF)) - self.mu * torch.log(detF) + 0.5 * self.mu * (trC - 3)
        return strainEnergy

    # ---------------------------------------------------------------------------------------
    # Purpose: calculate Neo-Hookean potential energy in 2D
    # ---------------------------------------------------------------------------------------
    def NeoHookean2D(self, u, x):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxy[:, 0].unsqueeze(1) + 1
        Fxy = duxdxy[:, 1].unsqueeze(1) + 0
        Fyx = duydxy[:, 0].unsqueeze(1) + 0
        Fyy = duydxy[:, 1].unsqueeze(1) + 1
        detF = Fxx * Fyy - Fxy * Fyx
        trC = Fxx ** 2 + Fxy ** 2 + Fyx ** 2 + Fyy ** 2
        strainEnergy = 0.5 * self.lam * (torch.log(detF) * torch.log(detF)) - self.mu * torch.log(detF) + 0.5 * self.mu * (trC - 2)
        return strainEnergy
