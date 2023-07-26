import numpy as np
from paddlets.metrics import MSE
from paddlets.logger import raise_if_not, Logger, raise_if

logger = Logger(__name__)
mse = MSE()


class GASearch(object):
    def __init__(self, gt, predict, cross_rate=0.5, mutation_rate=0.4):
        self.gt = gt
        self.predict = np.array(predict)
        self.pop_size = 8
        self.method_num = len(predict)
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.n_iters = 10
        self.early_stop = 5

    def get_fitness(self, pop):
        total_distance = np.empty(shape=(self.pop_size, ), dtype=np.float64)
        bs, predict_len, dim = self.gt.shape
        for i in range(self.pop_size):
            pop_i = pop[i]
            preds = self.predict[pop_i == 1]
            target_df = np.zeros(self.gt.shape)
            if preds.shape[0] > 1:
                for j in range(len(range(dim))):
                    meta = np.concatenate(
                        [
                            np.array(pred[:, :, j]).reshape(-1, 1)
                            for pred in preds
                        ],
                        axis=1)
                    y = np.mean(meta, axis=1)
                    target_df[:, :, j] = y.reshape(bs, predict_len)
                total_distance[i] = mse.metric_fn(self.gt, target_df)
            elif preds.shape[0] == 1:
                total_distance[i] = mse.metric_fn(self.gt, preds[0])
            else:
                total_distance[i] = 1000000

        fitness = np.exp(2.0 / total_distance)
        return fitness

    def select(self, fitness):
        idx = np.random.choice(
            np.arange(self.pop_size),
            size=self.pop_size,
            replace=True,
            p=fitness / fitness.sum())
        return idx

    def crossover_and_mutation(self, pop):
        new_pop = []
        old_pop = pop.copy()
        for child in pop.tolist():
            if np.random.rand() < self.cross_rate:
                mother = old_pop[np.random.randint(self.pop_size)]
                cross_points = np.random.randint(low=0, high=self.method_num)
                child[cross_points:] = mother[cross_points:]
            child = self.mutate(child)
            new_pop.append(child)
        return np.array(new_pop)

    def mutate(self, child):
        if np.random.rand() < self.mutation_rate:
            swap_point = np.random.randint(0, self.method_num)
            child[swap_point] = child[swap_point] ^ 1
        return child

    def evolve(self, pop, fitness):
        idx = self.select(fitness)
        pop = self.crossover_and_mutation(pop[idx])  # 交叉
        return pop

    def run(self):
        best_fitness = 0
        best_pop = []
        pop = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ])
        pop = np.random.randint(2, size=(self.pop_size, self.method_num))

        count = 0
        for generation in range(self.n_iters):
            fitness = self.get_fitness(pop)
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > best_fitness:
                best_fitness = fitness[best_idx]
                best_pop = pop[best_idx]
                count = 0
            elif fitness[best_idx] == best_fitness:
                count += 1
            logger.info('iter: {}, best_fitness: {}, best_pop: {}'.format(
                generation, 2.0 / np.log(best_fitness), best_pop))
            if count >= self.early_stop:
                break
            pop = self.evolve(pop, fitness)

        return best_pop


if __name__ == '__main__':

    predict = np.load('../../predict_5.npy')
    gt = np.load('../../gt_5.npy')[0]

    ga = GASearch(gt, predict)
    best_pop = ga.run()
    print(best_pop)
