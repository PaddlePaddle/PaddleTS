from paddle.optimizer.lr import LRScheduler


class Type1Decay(LRScheduler):
    def __init__(self, learning_rate, last_epoch=0, verbose=False):
        super(Type1Decay, self).__init__(learning_rate, last_epoch, verbose)
        self.last_lr = float(learning_rate)
        self.last_epoch = last_epoch

    def get_lr(self):
        # print(f'self.base_lr is {self.base_lr},  self.last_epoch  is {self.last_epoch}.')
        return self.base_lr * 0.5**((self.last_epoch - 1) // 1)
