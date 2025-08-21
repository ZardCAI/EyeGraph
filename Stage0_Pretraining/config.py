import os
import torch
from datetime import datetime
class Config:
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 优化参数
    batch_size = 640     # 根据显存调整
    max_epochs = 50
    max_lr = 1e-4      # 初始峰值学习率
    warmup_steps = 400      # 约2个epoch的warmup（假设batch_size=128, 数据集=10万样本）
    base_lr = 5e-5
    min_lr = 1e-6            # 最终学习率下限
    weight_decay = 0.2       # 保持强正则化
    betas = (0.9, 0.95)      # 调整动量参数
    
    # 实验管理
    result_dir = './result'
    checkpoint_dir = 'checkpoints'
    log_interval = 50
    save_interval = 5

    # Dataset
    root = 'FFA-IR'

    # Model
    vision_path = 'InternVL/pretrained/Mini-InternVL-Chat-4B-V1-5'
    llm_path = "InternVL/pretrained/EYE-Llama_gqa"
    
    @classmethod
    def setup(cls):
        os.makedirs(os.path.join(cls.result_dir, cls.checkpoint_dir), exist_ok=True)
        # TensorBoard日志目录
        cls.log_dir = os.path.join(cls.result_dir, 'runs', datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.makedirs(cls.log_dir, exist_ok=True)