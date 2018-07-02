from gym.spaces import Tuple


class Compound(Tuple):
    """
    A tuple (i.e., product) of simpler spaces.

    Exactly the same as Tuple except this supports indexing to access the sub-spaces.

    Example usage:
    --------------
    self.observation_space = spaces.Compound((spaces.Discrete(2), spaces.Discrete(3)))
    self.observation_space[0]  # Discrete(2)
    self.observation_space[1]  # Discrete(3)
    """
    def __getitem__(self, key):
        return self.spaces[key]

    def __repr__(self):
        return "Compound(" + ", ". join([str(s) for s in self.spaces]) + ")"