import torch

# 定义动作模型基本参数
class ActModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
    
    # 限定动作模型的输出范围为[-5,1]
    def forward(self, state):
        output = self.sequential(state)
        scaled_output = torch.tanh(output) * 3.0 - 2.0
        return scaled_output
    
    # 加载模型
    def load(self, path):
        self.load_state_dict(torch.load(path))