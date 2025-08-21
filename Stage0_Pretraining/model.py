import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from torch import nn
import numpy as np

class PretrainingClip(nn.Module):
    def __init__(self, vision_path, llm_path, vision_hidden_dim=1024, text_hidden_dim=4096, 
                 proj_dim=4096, dtype=torch.bfloat16):
        super().__init__()
        self.vision_model = self.init_visual(vision_path, dtype)
        self.tokenizer, self.text_model = self.init_text_llm(llm_path, dtype)
        self.vision_proj = nn.Linear(1024, 5, dtype=dtype)
        self.text_proj = nn.Linear(text_hidden_dim, proj_dim, dtype=dtype)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        self.dtype = dtype
        # CLIP风格的初始化
        nn.init.normal_(self.vision_proj.weight, std=vision_hidden_dim**-0.5)
        nn.init.normal_(self.text_proj.weight, std=text_hidden_dim**-0.5)

    def init_visual(self, vision_path, dtype):
        # vision part
        model = AutoModel.from_pretrained(
            vision_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        vision_model = model.vision_model
        return vision_model
    
    def init_text_llm(self, llm_path, dtype):
        tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        # tokenizer.pad_token = tokenizer.eos_token
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        for param in llm_model.parameters():
            param.requires_grad = False
        return tokenizer, llm_model.model

    def encode_image(self, image):
        # image = torch.randn((16, 3, 448, 448)).to(image.device)
        vit_embeds = self.vision_model(
                pixel_values=image.type(self.dtype),
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        vit_embeds = vit_embeds[:, 0, :]
        return self.vision_pr2oj(vit_embeds)

    def encode_text(self, text):
        device = self.text_model.device  # 自动获取模型所在设备
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        with torch.no_grad():  # 确保文本模型不计算梯度
            text_hidden_feature = self.text_model(**inputs).last_hidden_state

        attention_mask = inputs['attention_mask']
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(attention_mask.size(0), device=device)
        text_embeds = text_hidden_feature[batch_indices, seq_lengths]  # [batch_size, hidden_size]
        text_embeds = self.text_proj(text_embeds)
        return text_embeds

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
