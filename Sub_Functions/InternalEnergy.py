
import torch
import numpy as np

class InternalEnergy:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

    def Elastic2DGaussQuadMeta (self, u1 , u2 , x, dxdydz, shape, density , SENSITIVITY , Option , itr ):
        # SIMP
        simp_exponent = 3.
        if SENSITIVITY:
            density_SIMP = simp_exponent * torch.pow( density , simp_exponent - 1. )
        else:
            density_SIMP = torch.pow( density , simp_exponent )


        if not SENSITIVITY:
            Exx , Eyy , Exy , Sxx , Syy , Sxy , detJ = self.ElasticSE( u1, x, dxdydz, shape , density_SIMP )
            objective = ( Exx * Sxx + Eyy * Syy + 2 * Exy * Sxy ) * detJ * 0.5       

        else:
            if Option == 'Shear':
                Exx , Eyy , Exy , Sxx , Syy , Sxy , detJ = self.ElasticSE( u1, x, dxdydz, shape , density_SIMP )

                # E1212
                E1212 = ( Exx * Sxx + Eyy * Syy + 2 * Exy * Sxy )
                objective = E1212 * detJ * 0.5

            elif Option == 'Bulk':
                # Mode 1
                Exx1 , Eyy1 , Exy1 , Sxx1 , Syy1 , Sxy1 , detJ = self.ElasticSE( u1, x, dxdydz, shape , density_SIMP )
                # Mode 2
                Exx2 , Eyy2 , Exy2 , Sxx2 , Syy2 , Sxy2 , detJ = self.ElasticSE( u2, x, dxdydz, shape , density_SIMP )

                # E1111
                E1111 = ( Exx1 * Sxx1 + Eyy1 * Syy1 + 2 * Exy1 * Sxy1 )

                # E2222
                E2222 = ( Exx2 * Sxx2 + Eyy2 * Syy2 + 2 * Exy2 * Sxy2 )

                # E1122
                E1122 = ( Exx1 * Sxx2 + Eyy1 * Syy2 + 2 * Exy1 * Sxy2 )

                # E2211
                E2211 = ( Exx2 * Sxx1 + Eyy2 * Syy1 + 2 * Exy2 * Sxy1 )

                objective = ( E1111 + E2222 + E1122 + E2211 ) * detJ * 0.5


                # objective = ( np.power( 0.75 , itr + 1 ) * ( E1111 + E2222 ) - E1122 ) * detJ * 0.5


        if SENSITIVITY:
            return -objective , torch.sum( objective * density / simp_exponent ) 
        else:
            return torch.sum( objective )


    def ElasticSE(self, u, x, dxdydz, shape , density_SIMP ):
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
        
        S_xx_GP1= (self.E*(e_xx_GP1+ self.nu*e_yy_GP1)/(1-self.nu**2))*density_SIMP        
        S_yy_GP1= (self.E*(e_yy_GP1+ self.nu*e_xx_GP1)/(1-self.nu**2))*density_SIMP
        S_xy_GP1= (self.E*e_xy_GP1/(1+self.nu))*density_SIMP
                
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
                
        S_xx_GP2= (self.E*(e_xx_GP2+ self.nu*e_yy_GP2)/(1-self.nu**2))*density_SIMP        
        S_yy_GP2= (self.E*(e_yy_GP2+ self.nu*e_xx_GP2)/(1-self.nu**2))*density_SIMP
        S_xy_GP2= (self.E*e_xy_GP2/(1+self.nu))*density_SIMP
        
        
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
        
        S_xx_GP3= (self.E*(e_xx_GP3+ self.nu*e_yy_GP3)/(1-self.nu**2))*density_SIMP        
        S_yy_GP3= (self.E*(e_yy_GP3+ self.nu*e_xx_GP3)/(1-self.nu**2))*density_SIMP
        S_xy_GP3= (self.E*e_xy_GP3/(1+self.nu))*density_SIMP
                
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
        
        S_xx_GP4= (self.E*(e_xx_GP4+ self.nu*e_yy_GP4)/(1-self.nu**2))*density_SIMP        
        S_yy_GP4= (self.E*(e_yy_GP4+ self.nu*e_xx_GP4)/(1-self.nu**2))*density_SIMP
        S_xy_GP4= (self.E*e_xy_GP4/(1+self.nu))*density_SIMP
        
        # Strain at element
        Exx = ( e_xx_GP1 + e_xx_GP2 + e_xx_GP3 + e_xx_GP4 ) 
        Eyy = ( e_yy_GP1 + e_yy_GP2 + e_yy_GP3 + e_yy_GP4 )
        Exy = ( e_xy_GP1 + e_xy_GP2 + e_xy_GP3 + e_xy_GP4 )

        # Stress at element
        Sxx = ( S_xx_GP1 + S_xx_GP2 + S_xx_GP3 + S_xx_GP4 )
        Syy = ( S_yy_GP1 + S_yy_GP2 + S_yy_GP3 + S_yy_GP4 )
        Sxy = ( S_xy_GP1 + S_xy_GP2 + S_xy_GP3 + S_xy_GP4 )

        return Exx , Eyy , Exy , Sxx , Syy , Sxy , detJ


    def Elastic3DGaussQuad (self, u, x, dxdydz, shape, density , SENSITIVITY ):
        # SIMP
        simp_exponent = 3.
        if SENSITIVITY == 1:
            density_SIMP = simp_exponent * torch.pow( density , simp_exponent - 1. )
        else:
            density_SIMP = torch.pow( density , simp_exponent )

        order = [ 1 ,  shape[-1] , shape[0] , shape[1] ]
        Ux = torch.transpose(u[:, 0].reshape( order ), 2, 3)
        Uy = torch.transpose(u[:, 1].reshape( order ), 2, 3)
        Uz = torch.transpose(u[:, 2].reshape( order ), 2, 3)
        U = torch.cat( (Ux,Uy,Uz) , dim=0 )

        #        dim  z      y     x
        U_N1 = U[ : , :-1 , :-1 , :-1 ]
        U_N2 = U[ : , :-1 , :-1 , 1: ]
        U_N3 = U[ : , 1: , :-1 , 1: ]
        U_N4 = U[ : , 1: , :-1 , :-1 ]
        U_N5 = U[ : , :-1 , 1: , :-1 ]
        U_N6 = U[ : , :-1 , 1: , 1: ]
        U_N7 = U[ : , 1: , 1: , 1: ]
        U_N8 = U[ : , 1: , 1: , :-1 ]
        U_N = torch.stack( [ U_N1 , U_N2 , U_N3 , U_N4 , U_N5 , U_N6 , U_N7 , U_N8 ] ).double()

        # Compute constants
        detJ = dxdydz[0]*dxdydz[1]*dxdydz[2] / 8.
        Jinv = torch.zeros([3,3]).double()
        for i in range(3):
            Jinv[i,i] = 2. / dxdydz[i]

        grad2strain = torch.zeros([6,9]).double()
        grad2strain[0,0] = 1. # 11
        grad2strain[1,4] = 1. # 22
        grad2strain[2,8] = 1. # 33
        grad2strain[3,5] = 0.5; grad2strain[3,7] = 0.5 # 23
        grad2strain[4,2] = 0.5; grad2strain[4,6] = 0.5 # 13
        grad2strain[5,1] = 0.5; grad2strain[5,3] = 0.5 # 12 

        C_elastic = torch.zeros([6,6]).double()
        C_elastic[0,0] = 1. - self.nu; C_elastic[0,1] = self.nu; C_elastic[0,2] = self.nu
        C_elastic[1,0] = self.nu; C_elastic[1,1] = 1. - self.nu; C_elastic[1,2] = self.nu
        C_elastic[2,0] = self.nu; C_elastic[2,1] = self.nu; C_elastic[2,2] = 1. - self.nu
        C_elastic[3,3] = 1. - 2. * self.nu;
        C_elastic[4,4] = 1. - 2. * self.nu;
        C_elastic[5,5] = 1. - 2. * self.nu;
        C_elastic *= ( self.E / ( ( 1. + self.nu ) * ( 1. - 2. * self.nu ) ) )

        # Go through all integration pts
        strainEnergy_at_elem = torch.zeros_like( density_SIMP )
        if SENSITIVITY == -1:
            sss = density_SIMP.shape
            strain_at_elem = torch.zeros( [6,sss[0],sss[1],sss[2]] )
            stress_at_elem = torch.zeros( [6,sss[0],sss[1],sss[2]] )

        pts = [ -1. / np.sqrt(3) , 1. / np.sqrt(3) ]
        for x_ in pts:
            for y_ in pts:
                for z_ in pts:
                    # Shape grad in natural coords
                    B = torch.tensor([[-((y_ - 1)*(z_ - 1))/8, -((x_ - 1)*(z_ - 1))/8, -((x_ - 1)*(y_ - 1))/8],
                                [ ((y_ - 1)*(z_ - 1))/8,  ((x_ + 1)*(z_ - 1))/8,  ((x_ + 1)*(y_ - 1))/8],
                                [-((y_ - 1)*(z_ + 1))/8, -((x_ + 1)*(z_ + 1))/8, -((x_ + 1)*(y_ - 1))/8],
                                [ ((y_ - 1)*(z_ + 1))/8,  ((x_ - 1)*(z_ + 1))/8,  ((x_ - 1)*(y_ - 1))/8],
                                [ ((y_ + 1)*(z_ - 1))/8,  ((x_ - 1)*(z_ - 1))/8,  ((x_ - 1)*(y_ + 1))/8],
                                [-((y_ + 1)*(z_ - 1))/8, -((x_ + 1)*(z_ - 1))/8, -((x_ + 1)*(y_ + 1))/8],
                                [ ((y_ + 1)*(z_ + 1))/8,  ((x_ + 1)*(z_ + 1))/8,  ((x_ + 1)*(y_ + 1))/8],
                                [-((y_ + 1)*(z_ + 1))/8, -((x_ - 1)*(z_ + 1))/8, -((x_ - 1)*(y_ + 1))/8]]).double()
                    
                    # Convert to physical gradient
                    B_physical = torch.matmul( B , Jinv ).double()
                    dUx = torch.einsum( 'ijkl,iq->qjkl' , U_N[:,0,:,:,:] , B_physical )
                    dUy = torch.einsum( 'ijkl,iq->qjkl' , U_N[:,1,:,:,:] , B_physical )
                    dUz = torch.einsum( 'ijkl,iq->qjkl' , U_N[:,2,:,:,:] , B_physical )
                    dU = torch.cat( (dUx,dUy,dUz) , dim=0 )

                    # Strain [ 11 , 22 , 33 , 23 , 13 , 12 ]
                    eps = torch.einsum( 'qi,ijkl->qjkl' , grad2strain , dU )

                    # Stress [ 11 , 22 , 33 , 23 , 13 , 12 ]
                    Cauchy = torch.einsum( 'qi,ijkl->qjkl' , C_elastic , eps )

                    if SENSITIVITY == -1:
                        strain_at_elem += eps * 1. * detJ
                        stress_at_elem += Cauchy * density_SIMP * 1. * detJ

                    # Shear stresses need to be counted twice due to symmetry
                    Cauchy[3:,:,:,:] *= 2.
                    SE = 0.5 * torch.einsum( 'ijkl,ijkl->jkl' , Cauchy , eps ) 

                    # Scaled by design density
                    strainEnergy_at_elem += SE * density_SIMP * 1. * detJ

        if SENSITIVITY == 1:
            return -strainEnergy_at_elem , torch.sum( strainEnergy_at_elem * density / simp_exponent ) 
        elif SENSITIVITY == 0:
            return torch.sum( strainEnergy_at_elem )
        elif SENSITIVITY == -1:
            return strain_at_elem , stress_at_elem


    def Elastic2DGaussQuad (self, u, x, dxdydz, shape, density , SENSITIVITY ):
              
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

        # SIMP
        simp_exponent = 3.
        if SENSITIVITY:
            density_SIMP = simp_exponent * torch.pow( density , simp_exponent - 1. )
        else:
            density_SIMP = torch.pow( density , simp_exponent )


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
        
        S_xx_GP1= (self.E*(e_xx_GP1+ self.nu*e_yy_GP1)/(1-self.nu**2))*density_SIMP        
        S_yy_GP1= (self.E*(e_yy_GP1+ self.nu*e_xx_GP1)/(1-self.nu**2))*density_SIMP
        S_xy_GP1= (self.E*e_xy_GP1/(1+self.nu))*density_SIMP
        
        strainEnergy_GP1= (e_xx_GP1*S_xx_GP1+ e_yy_GP1*S_yy_GP1+ 2*e_xy_GP1*S_xy_GP1)
        
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
        
        S_xx_GP2= (self.E*(e_xx_GP2+ self.nu*e_yy_GP2)/(1-self.nu**2))*density_SIMP        
        S_yy_GP2= (self.E*(e_yy_GP2+ self.nu*e_xx_GP2)/(1-self.nu**2))*density_SIMP
        S_xy_GP2= (self.E*e_xy_GP2/(1+self.nu))*density_SIMP
        
        strainEnergy_GP2= (e_xx_GP2*S_xx_GP2+ e_yy_GP2*S_yy_GP2+ 2*e_xy_GP2*S_xy_GP2)
        
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
        
        S_xx_GP3= (self.E*(e_xx_GP3+ self.nu*e_yy_GP3)/(1-self.nu**2))*density_SIMP        
        S_yy_GP3= (self.E*(e_yy_GP3+ self.nu*e_xx_GP3)/(1-self.nu**2))*density_SIMP
        S_xy_GP3= (self.E*e_xy_GP3/(1+self.nu))*density_SIMP
        
        strainEnergy_GP3= (e_xx_GP3*S_xx_GP3+ e_yy_GP3*S_yy_GP3+ 2*e_xy_GP3*S_xy_GP3)
        
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
        
        S_xx_GP4= (self.E*(e_xx_GP4+ self.nu*e_yy_GP4)/(1-self.nu**2))*density_SIMP        
        S_yy_GP4= (self.E*(e_yy_GP4+ self.nu*e_xx_GP4)/(1-self.nu**2))*density_SIMP
        S_xy_GP4= (self.E*e_xy_GP4/(1+self.nu))*density_SIMP
        
        strainEnergy_GP4= (e_xx_GP4*S_xx_GP4+ e_yy_GP4*S_yy_GP4+ 2*e_xy_GP4*S_xy_GP4)

        # Strain energy at element
        strainEnergy_at_elem = ( strainEnergy_GP1 +strainEnergy_GP2 +strainEnergy_GP3 +strainEnergy_GP4 ) * detJ * 0.5
        
        if SENSITIVITY:
            return -strainEnergy_at_elem , torch.sum( strainEnergy_at_elem * density / simp_exponent ) 
        else:
            return torch.sum( strainEnergy_at_elem )