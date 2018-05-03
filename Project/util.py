import numpy as np

class EpsilonGreedy(object):
    """Helper class for an epsilon greedy stepper."""

    def __init__(self, start, end, steps):
        self.stepper = lambda n: start - n * (start-end)/steps
        self.n = 0

        self.start = start
        self.end = end
        self.steps = steps

    def get_eps(self):
        if self.n >= self.steps:
            return self.end
        else:
            eps = self.stepper(self.n)
            self.n += 1

            return eps

    def reset(self):
        self.n = 0