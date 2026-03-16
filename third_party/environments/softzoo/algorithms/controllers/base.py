import torch
import torch.nn as nn


class Base(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Base, self).__init__()
        self.optim = None

    def reset(self):
        raise NotImplementedError

    def forward(self, s, inp):
        raise NotImplementedError

    def update(self, grad, retain_graph=False):
        all_act = torch.stack(self.all_act)
        grad = torch.stack(grad)
        grad = grad.to(all_act) # make sure they are of the same type

        self.optim.zero_grad()
        all_act.backward(gradient=grad, retain_graph=retain_graph)
        self.optim.step()

    def save_checkpoint(self, filepath):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
