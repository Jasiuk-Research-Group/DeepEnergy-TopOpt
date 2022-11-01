import torch


class IntegrationLoss:
    def __init__(self, numIntType, dim):
        # print("Constructor: IntegrationLoss ", numIntType, " in ", dim, " dimension ")
        self.type = numIntType
        self.dim = dim

    def lossInternalEnergy(self, f, x=None, dx=1.0, dy=1.0, dz=1.0, shape=None):
        # Change return function here 
        return self.approxIntegration(f, x, dx, dy, dz, shape)

    def lossExternalEnergy(self, f, x=None, dx=1.0, dy=1.0, dz=1.0, shape=None):
        if self.type == 'trapezoidal':
            # print("Trapezoidal rule")
            if self.dim == 2:
                if x is not None:
                    print("x is not None")
                    return self.trapz1D(f, x=x)
                else:
                    return self.trapz1D(f, dx=dx)
            if self.dim == 3:
                if x is not None:
                    return self.trapz2D(f, xy=x, shape=shape)
                else:
                    return self.trapz2D(f, dx=dx, dy=dy, shape=shape)
                
                
    def approxIntegration(self, f, x=None, dx=1.0, dy=1.0, dz=1.0, shape=None):
        y = f.reshape(shape[0], shape[1])
        axis=-1
        
        nd = y.ndimension()
        slice1 = [slice(None)] * nd
        slice2 = [slice(None)] * nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        
        a1= y[:(shape[0]-1)][tuple(slice2)]
        a2= y[1:shape[0]][tuple(slice2)]
        a3= y[0:(shape[0]-1)][tuple(slice1)]
        a4= y[1:shape[0]][tuple(slice1)]
        
        b1= (a1+a2+a3)/6*dx*dy
        b2= (a3+a4+a2)/6*dx*dy
        
        b=b1+b2
        c= torch.sum(b)
        
        return c
    
    
    # def approxIntegration(self, f, x=None, dx=1.0, dy=1.0, dz=1.0, shape=None):
    #     if self.type == 'trapezoidal':
    #         # print("Trapezoidal rule")
    #         if self.dim == 1:
    #             if x is not None:
    #                 print("x is not None")
    #                 return self.trapz1D(f, x=x)
    #             else:
    #                 return self.trapz1D(f, dx=dx)
    #         if self.dim == 2:
    #             if x is not None:
    #                 print("x is not None")
    #                 return self.trapz2D(f, xy=x, shape=shape)
    #             else:
    #                 return self.trapz2D(f, dx=dx, dy=dy, shape=shape)
    #         if self.dim == 3:
    #             if x is not None:
    #                 return self.trapz3D(f, xyz=x, shape=shape)
    #             else:
    #                 return self.trapz3D(f, dx=dx, dy=dy, dz=dz, shape=shape)


    def trapz1D(self, y, x=None, dx=1.0, axis=-1):
        y1D = y.flatten()
        if x is not None:
            print("x is not None")
            x1D = x.flatten()
            return self.trapz(y1D, x1D, dx=dx, axis=axis)
        else:
            return self.trapz(y1D, dx=dx)

    def trapz2D(self, f, xy=None, dx=None, dy=None, shape=None):
        f2D = f.reshape(shape[0], shape[1])
        if dx is None and dy is None:
            x = xy[:, 0].flatten().reshape(shape[0], shape[1])
            y = xy[:, 1].flatten().reshape(shape[0], shape[1])
            return self.trapz(self.trapz(f2D, y[0, :]), x[:, 0])
        else:
            return self.trapz(self.trapz(f2D, dx=dy), dx=dx)

    def trapz3D(self, f, xyz=None, dx=None, dy=None, dz=None, shape=None):
        f3D = f.reshape(shape[0], shape[1], shape[2])
        if dx is None and dy is None and dz is None:
            print("dxdydz - trapz3D - Need to implement !!!")
        else:
            return self.trapz(self.trapz(self.trapz(f3D, dx=dz), dx=dy), dx=dx)


    def trapz(self, y, x=None, dx=1.0, axis=-1):
        # y = np.asanyarray(y)
        if x is None:
            d = dx
        else:
            d = x[1:] - x[0:-1]
            # reshape to correct shape
            shape = [1] * y.ndimension()
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        nd = y.ndimension()
        slice1 = [slice(None)] * nd
        slice2 = [slice(None)] * nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        ret = torch.sum(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
        return ret

