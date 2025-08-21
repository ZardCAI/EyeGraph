import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import os
from util import compute_metrics, save_checkpoint, get_custom_scheduler
from model import PretrainingClip
from dataset import ImageTextDataset
from tensorboardX import SummaryWriter
from config import Config
from tqdm import tqdm  # 添加tqdm导入

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
   
def train_stage0():
    # 初始化配置
    config = Config
    config.setup()
    
    writer = SummaryWriter(log_dir=config.log_dir)

    # 数据集准备
    dataset = ImageTextDataset(root=config.root)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    # 模型初始化
    model = PretrainingClip(
        vision_path=config.vision_path,
        llm_path=config.llm_path
    ).to(config.device)

    # 检查可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = AdamW(    
        model.parameters(),
        lr=config.base_lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98)
    )

    # 创建学习率调度器
    total_steps = len(dataloader) * config.max_epochs
    scheduler = get_custom_scheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        max_lr=config.max_lr,
        base_lr=config.base_lr,
        min_lr=config.min_lr
    )
        
    # 损失函数
    def clip_loss(logits_per_image, logits_per_text):
        # print('=====================================')
        # print(logits_per_image)
        labels = torch.arange(logits_per_image.size(0), 
                            device=logits_per_image.device)
        loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
        return (loss_img + loss_txt) / 2
    
    # 训练循环
    global_step = 0
    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        
        progress = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch}",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            dynamic_ncols=True
        )

        for batch_idx, (images, texts) in progress:
            # 数据准备
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LR', current_lr, global_step)
            # print(f'Learning Rate: {current_lr}')

            # if batch_idx % 5 == 0:  # 每100个batch检查一次
            #     print(f"Logit scale: {model.logit_scale.item():.4f}")  # 应该逐渐变化
            #     # 获取投影层参数
            #     vision_proj_weight = model.vision_proj.weight.detach()
            #     text_proj_weight = model.text_proj.weight.detach()
            #     print('====================================')
            #     print(vision_proj_weight)
            #     print('====================================')
            #     print(text_proj_weight)


            images = images.to(config.device)

            logits_per_image, logits_per_text = model(images, texts)
            loss = clip_loss(logits_per_image, logits_per_text)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()  # 更新学习率

            optimizer.zero_grad()
            
            batch_size = images.size(0)
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

            progress.set_postfix({
                'loss': f"{loss.item():.4f}",  # 当前batch的loss
                'avg_loss': f"{(epoch_loss/total_samples):.4f}"  # 平均loss
            })
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1


        
        if (epoch + 1) % config.save_interval == 0:
            avg_loss = epoch_loss / total_samples
            model.eval()
            with torch.no_grad():
                top1_correct = 0
                top5_correct = 0
                eval_samples = 0
                
                for images, texts in tqdm(dataloader, desc=f"Eval Epoch {epoch}"):
                    images = images.to(config.device)
                    logits_per_image, _ = model(images, texts)
                    
                    # 仅需计算image到text的准确率
                    batch_size = images.size(0)
                    top1, top5 = compute_metrics(logits_per_image, None)
                    top1_correct += top1
                    top5_correct += top5
                    eval_samples += batch_size
            
            # 计算最终指标
            top1_acc = top1_correct / eval_samples * 100
            top5_acc = top5_correct / eval_samples * 100
            
            # 记录TensorBoard
            writer.add_scalar('Accuracy/top1', top1_acc, epoch)
            writer.add_scalar('Accuracy/top5', top5_acc, epoch)
            
            # 打印epoch总结
            print(f"\nEpoch {epoch} Summary:")
            print(f"Avg Loss: {avg_loss:.4f} | Top1: {top1_acc:.2f}% | Top5: {top5_acc:.2f}%")

        # 保存checkpoint
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, config)

    
    # 保存最终模型
    final_state = {
        k: v for k, v in model.state_dict().items() 
        if any([
            k.startswith('vision_model'),
            k.startswith('vision_proj'),
            k.startswith('text_proj'),
            k == 'logit_scale'
        ])
    }
    torch.save(final_state, os.path.join(config.checkpoint_dir, 'final_model.pt'))
    
    writer.close()  # 关闭TensorBoard写入器
    print("Training completed!")

if __name__ == "__main__":
    train_stage0()