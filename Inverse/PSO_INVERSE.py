import torch

class ParticleSwarmOptimizer:
    def __init__(self, num_particles, dimensions, bounds):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.x = torch.empty(num_particles, dimensions)
        self.v = torch.zeros(num_particles, dimensions)
        
        # 初始化位置
        for dim in range(dimensions):
            self.x[:, dim] = torch.rand(num_particles) * (bounds[dim][1] - bounds[dim][0]) + bounds[dim][0]

        self.p_best = self.x.clone()
        self.g_best = self.x[self.objective_function(self.x).argmin()].clone()

        # 初始化历史记录列表
        self.positions_history = [self.x.clone()]
        self.objective_history = [self.objective_function(self.x).clone()]

    def objective_function(self, x):
        # 调用外部定义的目标函数
        return self.objective_func(x)

    def optimize(self, num_iterations, c1=1.5, c2=2.0, w=0.9):
        for i in range(num_iterations):

           
            r1 = torch.rand_like(self.x)
            r2 = torch.rand_like(self.x)
            self.v = w * self.v + c1 * r1 * (self.p_best - self.x) + c2 * r2 * (self.g_best - self.x)
            self.x += self.v

            # 更新个体最佳位置和全局最佳位置
            objective_values = self.objective_function(self.x)
            better_mask = objective_values < self.objective_function(self.p_best)
            self.p_best[better_mask] = self.x[better_mask]
            current_best_idx = objective_values.argmin()
            current_best_value = objective_values[current_best_idx]
            if current_best_value < self.objective_function(self.g_best):
                self.g_best = self.x[current_best_idx].clone()

            # 记录当前位置和目标函数值
            self.positions_history.append(self.x.clone())
            self.objective_history.append(objective_values.clone())
            print(f"step{i}",objective_values.shape) 

        return self.g_best, self.objective_function(self.g_best)

    def print_history(self):
        print("Objective Function Values Over Iterations:")
        for idx, values in enumerate(self.objective_history):
            print(f"Step {idx}: {values}")

# 示例代码
bounds = [(-5, 5), (-10, 10)]  # 为每个维度设置不同的界限
optimizer = ParticleSwarmOptimizer(num_particles=10000, dimensions=2, bounds=bounds)
g_best, min_value = optimizer.optimize(num_iterations=2000)
print(f"Global best position: {g_best}")
print(f"Minimum value: {min_value}")
optimizer.print_history()
