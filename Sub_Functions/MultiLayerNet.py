import rff
import torch

class MultiLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out,act_func, CNN_dev, rff_dev, N_Layers):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        H=int(H)
        super(MultiLayerNet, self).__init__()
        
        self.encoding = rff.layers.GaussianEncoding(sigma=rff_dev, input_size=D_in, encoded_size=H//2)
        
        N_Layers=int(N_Layers)

        ## Define loop to automate Layer definition  
        self.linear = torch.nn.ModuleList()

        for ii in range(N_Layers):
            if ii==0:
                self.linear.append(torch.nn.Linear(D_in, H))
                torch.nn.init.constant_(self.linear[ii].bias, 0.)
                torch.nn.init.normal_(self.linear[ii].weight, mean=0, std=CNN_dev)
            elif ii==(N_Layers-1):
                self.linear.append(torch.nn.Linear(H, D_out))
                torch.nn.init.constant_(self.linear[ii].bias, 0.)
                torch.nn.init.normal_(self.linear[ii].weight, mean=0, std=CNN_dev)
            else:
                self.linear.append(torch.nn.Linear(H,H))
                torch.nn.init.constant_(self.linear[ii].bias, 0.)
                torch.nn.init.normal_(self.linear[ii].weight, mean=0, std=CNN_dev)
        

    def forward(self, x,N_Layers,act_fn):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        activation_fn = getattr(torch,act_fn)       
        y_auto = []

        for ii in range(N_Layers):
            if ii==0:
                y_auto.append(self.encoding(x))
            elif ii==(N_Layers-1):
                y_auto.append(self.linear[-1](y_auto[-1]))
            else:
                y_auto.append(activation_fn(self.linear[ii](y_auto[ii-1])))
        
        return y_auto[-1]
