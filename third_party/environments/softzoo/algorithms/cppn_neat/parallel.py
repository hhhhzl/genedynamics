"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
import os
import pickle
from inspect import isfunction
import numpy as np
import torch
import torch.multiprocessing as multip
from cmaes import CMA


class Worker(multip.Process):
    EVAL_GENOME_FITNESS = 1
    EVAL_GENOME_CONSTRAINT = 2
    EXIT = 3

    def __init__(self, cls, *args, **kwargs):
        multip.Process.__init__(self)
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self.pipe, self.worker_pipe = multip.Pipe()
        self.daemon = True
        self.start()

    def wait(self):
        self.pipe.recv()

    def run(self):
        if isfunction(self.cls):
            func = self.cls
        else:
            func = self.cls(*self.args, **self.kwargs)
        self.worker_pipe.send('START')

        ans = None
        while True:
            op, data = self.worker_pipe.recv()
            if op == self.EVAL_GENOME_FITNESS:
                ans = func.evaluate_genome_fitness(*data)
                self.worker_pipe.send(ans)

            elif op == self.EVAL_GENOME_CONSTRAINT:
                ans = func.evaluate_genome_constraint(*data)
                self.worker_pipe.send(ans)

            elif op == self.EXIT:
                self.worker_pipe.close()
                return

    def close(self):
        self.pipe.send([self.EXIT, None])
        self.pipe.close()


class ParallelEvaluator(object):
    def __init__(self, num_workers, cls, cls_kwargs=dict()):
        self.num_workers = num_workers
        if self.num_workers > 0:
            self.workers = []
            for _ in range(self.num_workers):
                worker = Worker(cls, **cls_kwargs)
                self.workers.append(worker)

            [worker.wait() for worker in self.workers] # make sure all started
        else:
            self.local_worker = cls(**cls_kwargs)

        self.update_controller = cls_kwargs['args'].update_controller
        if self.update_controller:
            if self.num_workers > 0:
                controller = self.workers[0].cls.controller
            else:
                controller = self.local_worker.controller
            controller_params = [v for v in controller.parameters()]
            self.controller_params_shape = [v.shape for v in controller_params]
            controller_params = torch.cat([v.flatten() for v in controller_params]).data.numpy()
            self.controller_optimizer = CMA(mean=controller_params, sigma=cls_kwargs['args'].cmaes_sigma,
                                            population_size=cls_kwargs['args'].pop_size)

            self.out_dir = cls_kwargs['args'].out_dir

    def sample_controller(self, controller, controller_params_fl=None):
        if controller_params_fl is None:
            controller_params_fl = self.controller_optimizer.ask()
        controller_params = []
        offset = 0
        for param_shape in self.controller_params_shape:
            step = np.prod(param_shape)
            controller_params.append(controller_params_fl[offset:offset+step])
            offset += step
        for i, param in enumerate(controller.parameters()):
            param.data = torch.from_numpy(controller_params[i]).to(param).reshape(param.data.shape)

        return controller_params_fl

    def evaluate_fitness(self, genomes, config, generation, render=False):
        if self.update_controller:
            controller_params_samples = []

        if self.num_workers > 0:
            n_runs = int(np.ceil(len(genomes) / self.num_workers))
            assert not self.update_controller, 'Haven\'t figure how to handle async controller update'
            for i in range(n_runs):
                for j, worker in enumerate(self.workers):
                    genome_id = i * self.num_workers + j
                    if genome_id < len(genomes):
                        _, genome = genomes[genome_id]
                        data = (genome, config, genome_id, generation, render)
                        worker.pipe.send([worker.EVAL_GENOME_FITNESS, data])

                for j, worker in enumerate(self.workers):
                    genome_id = i * self.num_workers + j
                    if genome_id < len(genomes):
                        _, genome = genomes[genome_id]
                        genome.fitness = worker.pipe.recv()
        else:
            worker = self.local_worker
            for i, (_, genome) in enumerate(genomes):
                if self.update_controller:
                    controller_params = self.sample_controller(self.local_worker.controller)
                    controller_params_samples.append(controller_params)
                data = (genome, config, i, generation, render)
                genome.fitness = worker.evaluate_genome_fitness(*data)

        if self.update_controller and genomes[0][0] is not None: # only rendering best if genome_id is none
            all_fitness = [genome.fitness for _, genome in genomes]
            all_solutions = [(v1, -v2) for v1, v2 in zip(controller_params_samples, all_fitness)] # NOTE: CMAES takes in loss instead of fitness
            if len(all_solutions) != self.controller_optimizer.population_size: # HACK: to handle varying number of genomes
                if True: # sorted by top-fitness
                    topk_idcs = np.argsort(all_fitness)[::-1][:self.controller_optimizer.population_size]
                    all_solutions = [v for i, v in enumerate(all_solutions) if i in topk_idcs]
                else: # random
                    all_solutions = all_solutions[:self.controller_optimizer.population_size]
            self.controller_optimizer.tell(all_solutions)

            # save checkpoint
            out_dir = os.path.join(self.out_dir, f'generation_{generation}', 'control')
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'optimizer.pkl'), 'wb') as f:
                pickle.dump(self.controller_optimizer, f)
            with open(os.path.join(out_dir, 'optimizer_raw.pkl'), 'wb') as f:
                pickle.dump({
                    'mean': self.controller_optimizer._mean,
                    'sigma': self.controller_optimizer._sigma,
                    'population_size': self.controller_optimizer.population_size,
                }, f)

            best_idx = np.argsort(all_fitness)[::-1][0]
            best_params = controller_params_samples[best_idx]
            if self.num_workers > 0:
                raise NotImplementedError
            else:
                controller = self.local_worker.controller
                self.sample_controller(controller, best_params)
            controller.save_checkpoint(os.path.join(out_dir, 'th_state_dict.pt'))

    def evaluate_constraint(self, genomes, config, generation):
        validity_all = []
        if self.num_workers > 0:
            for i, (_, genome) in enumerate(genomes):
                worker_id = i % self.num_workers
                worker = self.workers[worker_id]
                data = (genome, config, i, generation)
                worker.pipe.send([worker.EVAL_GENOME_CONSTRAINT, data])
                validity = worker.pipe.recv()
                validity_all.append(validity)
        else:
            worker = self.local_worker
            for i, (_, genome) in enumerate(genomes):
                data = (genome, config, i, generation)
                validity = worker.evaluate_genome_constraint(*data)
                validity_all.append(validity)

        return validity_all
