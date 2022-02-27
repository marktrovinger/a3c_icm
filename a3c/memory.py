class Memory():
    def __init__(self):

        self.log_probs = []
        self.values = []
        self.rewards = []

    def clear_memory(self):
        # clears the memory for the replay
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()

    def store_memory(self, log_prob, values, rewards):
        # stores the memory
        self.log_probs.append(log_prob)
        self.values.append(values)
        self.rewards.append(rewards)
    
    def retrievaluese_memory(self):
        return self.log_probs, self.values, self.rewards