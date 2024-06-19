import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import autograd

class I2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, K, norm_layer=nn.BatchNorm1d, act=nn.ReLU):
        super(I2Layer,self).__init__()
        self.K=K
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.norm_layer=norm_layer(out_dim) if norm_layer else None
        self.act=act() if act else None

        self.learners=nn.ModuleList([nn.Linear(in_dim,out_dim,bias=True) for i in range(K)])

    def forward(self,x,g):
        """
        @input:
            x.shape = (bsz,in,h*w)
            g.shape = (bsz,K,h*w)
        @return:
            (bsz,out,h*w)
        """
        bsz,in_dim,hw=x.shape

        x=x.permute(0,2,1).contiguous() # (bsz,h*w,in)
        g=g.permute(0,2,1).contiguous() # (bsz,h*w,K)
        
        x=torch.cat([learner(x).unsqueeze(-1) for learner in self.learners],dim=-1) # (bsz,h*w,out,K)
        x=x.permute(0,1,3,2).contiguous() # (bsz,h*w,K,out)
        x=torch.sum(x*g.unsqueeze(-1),dim=-2) # (bsz,h*w,out)
        x=x.permute(0,2,1).contiguous() # (bsz,out,h*w)
        
        if self.norm_layer:
            x=self.norm_layer(x)
        if self.act:
            x=self.act(x)
        return x
    
    

class I2Net(nn.Module):
    def __init__(self, in_dim, planes, K, norm_layer=nn.BatchNorm1d, act=nn.ReLU, gate_layer=1):
        super(I2Net,self).__init__()
        self.in_dim=in_dim
        self.planes=planes
        self.K=K
        self.norm_layer=norm_layer
        self.act=act

        all_planes=[in_dim]+self.planes
        self.I2Layers=[I2Layer(all_planes[i],
                               all_planes[i+1],K,norm_layer,
                               act) for i in range(len(all_planes)-2)
                       ] + [I2Layer(all_planes[-2],all_planes[-1],K,None,None)] # no act in last layer
        self.I2Layers=nn.ModuleList(self.I2Layers)
        
        if gate_layer == 2:
            self.gate_net = nn.Sequential(
            nn.Conv1d(in_dim, 256, 1), norm_layer(256), nn.ReLU(), 
            nn.Conv1d(256, 128, 1), norm_layer(128), nn.ReLU(),
            nn.Conv1d(128, K, 1)
            )
        elif gate_layer == 1:
            self.gate_net = nn.Sequential(
            nn.Conv1d(in_dim, 128, 1), norm_layer(128), nn.ReLU(), 
            nn.Conv1d(128, K, 1)
            )               
        else:
            self.gate_net = nn.Sequential(
            nn.Conv1d(in_dim, 512, 1), norm_layer(512), nn.ReLU(), 
            nn.Conv1d(512, 256, 1), norm_layer(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), norm_layer(128), nn.ReLU(),
            nn.Conv1d(128, K, 1)
            )
        

    def forward(self,x,return_gate=False):
        """
        @input:
            x.shape = (bsz,self.in_dim,h*w)
        @return:
            (bsz,self.planes[-1],h*w)
        """

        gate=F.softmax(self.gate_net(x),dim=1) # (bsz,K,h*w)
        for layer in self.I2Layers:
            x=layer(x,gate)
            
        if return_gate:
            return x,gate
        else:
            return x
        
class GateShapingLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,gate):
        """
        @input:
            gate: (bsz,K,h,w)
        """
        # converged to 0.
        gate_instance_sharpening=-torch.mean(torch.sum(gate*torch.log(gate),dim=1))
        gate=torch.mean(gate,dim=(0,2,3))
        # converged to -1.3863
        gate_expectation_smoothing=torch.sum(gate*torch.log(gate))
        alpha_ges=0.5
        return gate_instance_sharpening+gate_expectation_smoothing*alpha_ges
    


class OrthGradLoss(nn.Module):
    def __init__(self,K):
        super().__init__()
        self.K=K
        self.EPS=1e-8

    def forward(self,seg_loss,layers):
        loss=0.
        for layer in layers:
            loss=loss+self.forward_layer(seg_loss,layer.learners)
        return loss
    
    def forward_layer(self,seg_loss,learners):
        weights=[learner.weight for learner in learners]
        grads = autograd.grad(
            outputs=seg_loss,
            inputs=weights,
            grad_outputs=torch.ones_like(seg_loss),
            create_graph=True,
            retain_graph=True,
            only_inputs=True) # K*(out,in)
        grads = torch.cat(grads,dim=0).view(self.K,-1) # (K,out*in)
        norm = torch.sqrt(torch.sum(torch.square(grads),dim=-1)) # (K)
        grads = grads/(self.EPS+torch.unsqueeze(norm,dim=1)) # (K,out*in)
        loss = torch.sum(torch.square(torch.matmul(
            grads,grads.permute(1,0).contiguous())*((torch.ones((self.K,self.K))-torch.eye(self.K)).to(seg_loss.device))))/2./self.K/(self.K-1)
        
        return loss

        