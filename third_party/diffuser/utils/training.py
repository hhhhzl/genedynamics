import os, pickle
import copy
import numpy as np
import torch
from tqdm.auto import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup

from .arrays import batch_to_device

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        train_test_split=1.0,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        lr_warmup_steps=1000,
        n_train_steps=1e5,
        n_steps_per_epoch=1000,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=1000,
        train_device='cuda',
        results_folder='./results',
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.save_freq = n_train_steps // 5

        self.n_train_steps = n_train_steps
        self.n_steps_per_epoch = n_steps_per_epoch
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.train_test_split = train_test_split
        if train_test_split == 1:
            self.train_dataloader = cycle(torch.utils.data.DataLoader(
                self.dataset, batch_size=train_batch_size, num_workers=2, shuffle=True, pin_memory=True
            ))
        else:
            n_train = int(train_test_split * len(self.dataset))
            n_test = len(self.dataset) - n_train
            train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [n_train, n_test])
            self.train_dataloader = cycle(torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, num_workers=2, shuffle=True, pin_memory=True
            ))
            self.test_dataloader = cycle(torch.utils.data.DataLoader(
                test_dataset, batch_size=train_batch_size, num_workers=2, shuffle=True, pin_memory=True
            ))
            self.best_test_loss = np.inf
        self.train_losses = []
        self.test_losses = []
        self.train_a0_losses = []
        self.test_a0_losses = []

        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=self.n_train_steps,
        )

        self.logdir = results_folder

        self.reset_parameters()
        self.step = 0

        self.device = train_device

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train_epoch(self, n_train_steps, epoch=0):        
        progress_bar = tqdm(total=n_train_steps)
        progress_bar.set_description(f"Epoch {epoch}")

        for step in range(n_train_steps):
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.train_dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step
                self.save(label)

            if self.step % self.log_freq == 0:
                self.train_losses.append([self.step, loss.item()])
                if 'a0_loss' in infos:
                    self.train_a0_losses.append([self.step, infos['a0_loss'].item()])

                if self.train_test_split < 1:
                    test_loss, test_a0_loss = self.test()
                    self.test_losses.append([self.step, test_loss])
                    self.test_a0_losses.append([self.step, test_a0_loss])
                    if test_loss < self.best_test_loss:
                        self.best_test_loss = test_loss
                        self.save_best()
                    if self.step == 0:
                        # if self.model.action_dim > 0:
                        print(f'Initial test loss: {test_loss:8.4f}, a0 loss: {test_a0_loss:8.4f}')
                        # else:
                            # print(f'Initial test loss: {test_loss:8.4f}')
                self.save_losses()
            
            # if self.train_test_split < 1:
            #     if 'a0_loss' in infos:
            #         logs = {"loss": loss.item(), "a0_loss": infos['a0_loss'].item(), "loss_test": test_loss, "a0_loss_test": test_a0_loss, "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step}
            #     else:
            #         logs = {"loss": loss.item(), "loss_test": test_loss, "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step}
            # else:
            #     if 'a0_loss' in infos:
            #         logs = {"loss": loss.item(), "a0_loss": infos['a0_loss'].item(), "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step}
            #     else:
            #         logs = {"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step}

            logs = {"loss": loss.item()}
            for l in ["diffusion_loss", "dyn_loss", "a0_loss", ]:
                if l in infos:
                    logs[l] = infos[l].item()
            if self.train_test_split < 1:
                logs["loss_test"] = test_loss
                if 'a0_loss' in infos:
                    logs["a0_loss_test"] = test_a0_loss
            logs["lr"] = self.lr_scheduler.get_last_lr()[0]
            logs["step"] = self.step

            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

            self.step += 1

    def train(self):
        n_epochs = int(self.n_train_steps // self.n_steps_per_epoch)
        for epoch in range(n_epochs):
            self.train_epoch(self.n_steps_per_epoch, epoch)

    def test(self, n_test=100):
        self.model.eval()   # Set the model to evaluation mode

        test_loss = 0
        test_a0_loss = 0
        with torch.no_grad():
            for step in range(n_test):
                batch = next(self.test_dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss /= self.gradient_accumulate_every
            
                test_loss += loss.item()
                test_a0_loss += infos['a0_loss'].item() if 'a0_loss' in infos else 0

            test_loss /= n_test
            test_a0_loss /= n_test

        return test_loss, test_a0_loss

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        # print(f'Saved model to {savepath}', flush=True)

    def save_best(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_best.pt')
        torch.save(data, savepath)
        # print(f'Saved best model to {savepath}', flush=True)

    def save_losses(self):
        data = {
            'training_losses': self.train_losses,
            'test_losses': self.test_losses,
            'training_a0_losses': self.train_a0_losses,
            'test_a0_losses': self.test_a0_losses,
        }
        savepath = os.path.join(self.logdir, 'losses.pkl')

        with open(savepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
