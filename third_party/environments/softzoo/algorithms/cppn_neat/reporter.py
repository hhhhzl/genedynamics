import os
import neat
import pickle
import numpy as np


class SaveResultReporter(neat.reporting.BaseReporter):
    def __init__(self, save_path, other_saved_data=dict()):
        super().__init__()
        self.save_path = save_path
        self.generation = None
        self.other_saved_data = other_saved_data

    def start_generation(self, generation):
        self.generation = generation
        save_path_design = os.path.join(self.save_path, f'generation_{generation}', 'design')
        os.makedirs(save_path_design, exist_ok=True)

    def post_evaluate(self, config, population, species, best_genome):
        save_path_ranking = os.path.join(self.save_path, f'generation_{self.generation}', 'output.txt')
        genome_id_list, genome_list = np.arange(len(population)), np.array(list(population.values()))
        sorted_idx = sorted(genome_id_list, key=lambda i: genome_list[i].fitness, reverse=True)
        genome_id_list, genome_list = list(genome_id_list[sorted_idx]), list(genome_list[sorted_idx])
        with open(save_path_ranking, 'w') as f:
            out = ''
            for genome_id, genome in zip(genome_id_list, genome_list):
                out += f'{genome_id}\t\t{genome.fitness}\n'
            f.write(out)
        
        save_path_gnome = os.path.join(self.save_path, f'generation_{self.generation}', 'design', 'genome.pkl')
        with open(save_path_gnome, 'wb') as f:
            data = dict(config=config, genome=best_genome)
            data.update(self.other_saved_data)
            pickle.dump(data, f)