import numpy as np

class Incremental_Timesteps():
    def __init__(self, F, T):
        self.F = F
        self.T = T

        mat = np.zeros((T, F))
        # 初始化最后一列，每个时间步的路径数为1
        for t in range(T):
            mat[t, F - 1] = 1
        # 从后往前数视频帧
        for f in range(F - 2, -1, -1):
            mat[T - 1, f] = 1
            for t in range(T - 2, -1, -1):
                mat[t, f] = mat[t + 1, f] + mat[t, f + 1]
        self.mat_s = mat

        mat = np.zeros((T, F))
        # 初始化第一列，每个时间步的路径数为1
        for t in range(T):
            mat[t, 0] = 1
        # 从前往后数视频帧
        for f in range(1, F):
            mat[0, f] = 1
            for t in range(1, T):
                mat[t, f] = mat[t - 1, f] + mat[t, f - 1]
        self.mat_e = mat

    def sample_step_sequence(self):
        preT = 0
        timesteps = np.zeros(self.F)

        for f in range(self.F):
            candidate_weights = self.mat_s[preT:, f]
            sum_weight = np.sum(candidate_weights)
            prob_sequence = candidate_weights / sum_weight
            cur_step = np.random.choice(range(preT, self.T), p=prob_sequence)
            timesteps[f] = cur_step
            preT = cur_step
        return timesteps

    def sample_stepseq_from_mid(self):
        timesteps = np.zeros(self.F)
        # 随机选择一个视频帧，对这个视频帧的时间步进行均匀采样
        curf = np.random.randint(self.F)
        timesteps[curf] = np.random.randint(self.T)
        
        # 生成之前视频帧的时间步, 之前视频帧的噪音逐步变小
        for f in range(curf - 1, -1, -1):
            # print(f, timesteps[f+1], self.mat_e)
            candidate_weights = self.mat_e[:int(timesteps[f+1]) + 1, f]
            sum_weight = np.sum(candidate_weights)
            prob_sequence = candidate_weights / sum_weight
            cur_step = np.random.choice(range(0, int(timesteps[f+1]) + 1), 
                                        p=prob_sequence)
            timesteps[f] = int(cur_step)
        
        # 生成之后视频帧的时间步
        for f in range(curf + 1, self.F):
            candidate_weights = self.mat_s[int(timesteps[f-1]):, f]
            sum_weight = np.sum(candidate_weights)
            prob_sequence = candidate_weights / sum_weight
            cur_step = np.random.choice(range(int(timesteps[f-1]), self.T), 
                                        p=prob_sequence)
            timesteps[f] = int(cur_step)
        
        return timesteps

