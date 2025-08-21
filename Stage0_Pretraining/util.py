import torch
from torch.optim.lr_scheduler import LambdaLR
import math
import os

def compute_metrics(logits, targets, topk=5):
    """
    计算top1和topk准确率
    logits: [batch_size, batch_size]
    targets: 正确索引应该是0到batch_size-1的序列
    """
    targets = torch.arange(logits.size(0), device=logits.device)
    
    # Top-1
    _, pred_top1 = logits.topk(1, dim=1)
    correct_top1 = (pred_top1 == targets.unsqueeze(1)).sum().item()
    
    # Top-k
    _, pred_topk = logits.topk(topk, dim=1)
    correct_topk = (pred_topk == targets.unsqueeze(1)).any(dim=1).sum().item()
    
    return correct_top1, correct_topk

def save_checkpoint(model, optimizer, epoch, config):
    # 只保存需要训练的部分
    state_dict = {
        k: v for k, v in model.state_dict().items() 
        if any([
            k.startswith('vision_model'),  # 视觉模型
            k.startswith('vision_proj'),   # 视觉投影
            k.startswith('text_proj'),     # 文本投影
            k == 'logit_scale'             # 温度参数
        ])
    }
    
    # 优化器状态也需要过滤
    opt_state = {
        'param_groups': optimizer.param_groups,
        'state': {
            k: v for k, v in optimizer.state_dict()['state'].items()
            if k in state_dict.keys()  # 只保留需要参数的优化状态
        }
    }
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': opt_state,
    }, os.path.join(config.result_dir, config.checkpoint_dir, f'clip_epoch{epoch}.pt'))

def get_custom_scheduler(optimizer, warmup_steps, total_steps, max_lr=1e-3, base_lr=1e-4, min_lr=1e-6):
    """
    正确版本的学习率调度器
    """
    def lr_lambda(current_step):
        # Warmup阶段：计算相对于base_lr的缩放系数
        if current_step < warmup_steps:
            alpha = current_step / warmup_steps
            # 从1.0增长到 max_lr/base_lr
            return 1.0 + (max_lr/base_lr - 1.0) * alpha
        
        # Cosine衰减阶段
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        # 从max_lr衰减到min_lr的相对系数
        return (cosine_decay * (max_lr - min_lr) + min_lr) / base_lr
    
    return LambdaLR(optimizer, lr_lambda)
    
