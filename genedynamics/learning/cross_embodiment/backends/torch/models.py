"""Torch CPU/CUDA numerical provider sharing the portable in_out contract."""
import torch
from ...models import TrainState, RegressionBatch


def forward(parameters, inputs):
    value = inputs.unsqueeze(0).expand(parameters[0][0].shape[0],-1,-1)
    for i,(weight,bias) in enumerate(parameters):
        value = torch.einsum("ebi,eio->ebo",value,weight)+bias[:,None,:]
        if i+1 != len(parameters):
            value = torch.tanh(value)
    return value


class TorchModel:
    def __init__(self,spec,device):
        if device.startswith(("gpu", "cuda")):
            device = device.replace("gpu","cuda")
            if not torch.cuda.is_available():
                raise RuntimeError("requested CUDA device unavailable; no CPU fallback")
        self.device,self.spec = torch.device(device),spec

    def _tensor(self,value):
        return torch.as_tensor(value,device=self.device,dtype=torch.float32)

    def initialize(self,seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        sizes = (self.spec.input_dim,*self.spec.hidden,self.spec.output_dim)
        parameters = tuple((torch.randn((self.spec.ensemble,n,m),generator=generator,device=self.device)*
                            (2./(n+m))**.5,torch.zeros((self.spec.ensemble,m),device=self.device))
                           for n,m in zip(sizes[:-1],sizes[1:]))
        zero = tuple(tuple(torch.zeros_like(x) for x in layer) for layer in parameters)
        return TrainState(parameters,zero,zero,torch.tensor(0,device=self.device))

    def predict(self,parameters,inputs):
        return forward(parameters,self._tensor(inputs))

    def train_step(self,state,batch):
        inputs,targets,weights = (self._tensor(x) for x in batch)
        parameters = tuple(tuple(x.detach().requires_grad_() for x in layer) for layer in state.parameters)
        error = forward(parameters,inputs)-targets[None]
        denominator = (self.spec.ensemble*error.shape[-1]*weights.sum()).clamp(min=1.)
        loss = (error.square()*weights[None,:,None]).sum()/denominator
        gradients = torch.autograd.grad(loss,tuple(x for layer in parameters for x in layer))
        flat_first = tuple(x for layer in state.first for x in layer)
        flat_second = tuple(x for layer in state.second for x in layer)
        step = state.step+1
        first = tuple(.9*m+.1*g for m,g in zip(flat_first,gradients))
        second = tuple(.999*v+.001*g.square() for v,g in zip(flat_second,gradients))
        flat_parameters = tuple(x for layer in parameters for x in layer)
        updated = tuple((p-self.spec.learning_rate*(m/(1-.9**step))/(torch.sqrt(v/(1-.999**step))+1e-8)).detach()
                        for p,m,v in zip(flat_parameters,first,second))
        paired = lambda xs: tuple((xs[i],xs[i+1]) for i in range(0,len(xs),2))
        active = weights.sum()>0
        choose = lambda new,old: tuple(tuple(torch.where(active,n,o) for n,o in zip(nl,ol))
                                       for nl,ol in zip(new,old))
        return TrainState(choose(paired(updated),state.parameters),choose(paired(first),state.first),
                          choose(paired(second),state.second),torch.where(active,step,state.step)),loss.detach()

    def train_chunk(self,state,batches):
        losses=[]
        for i in range(batches.inputs.shape[0]):
            state,loss = self.train_step(state,RegressionBatch(*(x[i] for x in batches)))
            losses.append(loss)
        return state,torch.stack(losses)

    def input_gradient(self,parameters,inputs,cotangent):
        value = self._tensor(inputs).detach().requires_grad_()
        return torch.autograd.grad((forward(parameters,value)*self._tensor(cotangent)).sum(),value)[0]

    def import_parameters(self,tensors):
        sizes=(self.spec.input_dim,*self.spec.hidden,self.spec.output_dim)
        expected={f"layer.{i}.{name}" for i in range(len(sizes)-1) for name in ("weight","bias")}
        if set(tensors)!=expected:
            raise ValueError("model layer manifest mismatch")
        layers=[]
        for i,(n,m) in enumerate(zip(sizes[:-1],sizes[1:])):
            w,b=tensors[f"layer.{i}.weight"],tensors[f"layer.{i}.bias"]
            if w.shape!=(self.spec.ensemble,n,m) or b.shape!=(self.spec.ensemble,m):
                raise ValueError("model layer shape mismatch")
            layers.append((self._tensor(w),self._tensor(b)))
        return tuple(layers)

    def synchronize(self,value):
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def export_tensors(tensors):
        return {k:v.detach().cpu().numpy() for k,v in tensors.items()}
