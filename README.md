# 使用load.py加载模型
# 模型的输入为 torch.tensor([acc, h, delta_v, v], dtype=torch.float32)，即跟随车加速度，两车距离，跟随车与前车速度差，前车速度
# 模型输出为跟随车加速度
 # 加载模型，path为上述模型文件的路径
    def load(self, path):
        self.load_state_dict(torch.load(path))
