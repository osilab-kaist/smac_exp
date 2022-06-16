import numpy as np


class RewardSchedule():
    def __init__(self, reward_alt_start_ratio=1, reward_alt_end_ratio=0,
        reward_alt_end_step=500000, reward_alt_method="linear"):

        self.start = reward_alt_start_ratio
        self.finish = reward_alt_end_ratio
        self.time_length = reward_alt_end_step
        self.delta = (self.start - self.finish) / self.time_length
        if self.start < self.finish:
            self.delta = (self.finish + self.start) / self.time_length
        self.method = reward_alt_method

    def eval(self, T):
        if self.method in ["linear", "constant"]:
            if self.start < self.finish:
                return min(self.finish, sefelf.start - self.delta * T)
            else:
                return max(self.finish, self.start - self.delta * T)

        elif self.method in ["step"]:
            if T < self.time_length:
                return self.start
            else:
                return self.finish

if __name__ == "__main__":
    alt_reward = RewardSchedule(1,0,1000000,"linear")
    print(alt_reward.eval(500000))