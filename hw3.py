import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern)
            pattern = pattern.reshape((self.size, 1))
            self.weights += np.dot(pattern, pattern.T)
            np.fill_diagonal(self.weights, 0)
        self.weights /= self.size

    def energy(self, state):
        return -0.5 * np.dot(np.dot(state.T, self.weights), state)

    def update(self, state, steps=100):
        for _ in range(steps):
            for neuron in range(self.size):
                s = np.dot(self.weights[neuron], state)
                state[neuron] = np.sign(s)

        return state



def create_pattern(letter):
    patterns = {
    '1': [-1, -1, -1, -1, -1,
          -1,  1, -1,  1, -1,
          -1, -1,  1, -1, -1,
          -1, -1,  1, -1, -1,
          -1, -1,  1, -1, -1,
          -1, -1,  1, -1, -1,
          -1, -1,  1, -1, -1,
          -1,  1,  1,  1, -1,
          -1, -1, -1, -1, -1],

    '2': [-1,  1,  1,  1, -1,
           1, -1, -1, -1,  1,
          -1, -1, -1, -1,  1,
          -1, -1, -1,  1, -1,
          -1, -1,  1, -1, -1,
          -1,  1, -1, -1, -1,
           1, -1, -1, -1, -1,
           1,  1,  1,  1,  1,
          -1, -1, -1, -1, -1],

    '3': [ 1,  1,  1,  1,  1,
          -1, -1, -1, -1,  1,
          -1, -1, -1,  1, -1,
          -1,  1,  1,  1, -1,
          -1, -1, -1,  1, -1,
          -1, -1, -1, -1,  1,
           1, -1, -1, -1,  1,
          -1,  1,  1,  1, -1,
          -1, -1, -1, -1, -1],

     '4': [-1, -1, -1, -1, -1,
          -1,  1, -1,  1, -1,
          -1,  1, -1,  1, -1,
           1, -1,  1, -1,  1,
           1,  1,  1,  1,  1,
          -1, -1, -1,  1, -1,
          -1, -1, -1,  1, -1,
          -1, -1, -1,  1, -1,
          -1, -1, -1, -1, -1]
    }
    return np.array(patterns[letter]).flatten()

def distort_pattern(pattern, noise_level=0.2):
    noisy_pattern = np.copy(pattern)
    num_distort = int(len(pattern) * noise_level)
    indices = np.random.choice(len(pattern), num_distort, replace=False)
    for idx in indices:
        noisy_pattern[idx] *= -1  # 將模式中的值轉換為相反的值，-1 變成 1，1 變成 -1
    return noisy_pattern

def display_pattern(pattern):
    for i in range(9):
        print(''.join(' ' if x == -1 else '█' for x in pattern[i * 5:(i + 1) * 5]))  


# Main simulation
np.random.seed(42)  
letters = [ '4']

network = HopfieldNetwork(45)
patterns = [create_pattern(letter) for letter in letters]
network.train(patterns)

for letter in letters:
    pattern = create_pattern(letter)
    distorted_pattern = distort_pattern(pattern, noise_level=0.5)

    print(f"Original {letter} pattern:")
    display_pattern(pattern.reshape(45))
    
    print(f"Distorted {letter} pattern:")
    display_pattern(distorted_pattern.reshape(45))

    recovered_pattern = network.update(distorted_pattern.copy())
    print(f"Recovered {letter} pattern:")
    display_pattern(recovered_pattern.reshape(45))
    print("\n")
