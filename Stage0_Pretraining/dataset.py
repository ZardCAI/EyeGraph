from PIL import Image
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

class ImageTextDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.caption_csv = os.path.join(root, 'ffa_train.csv')
        self.transform = transform or transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        captio_df = pd.read_csv(self.caption_csv)
        self.img_ids = captio_df['Img ID'].to_list()
        self.captions = captio_df['Caption'].to_list()
            
    def __len__(self):
        return len(self.img_ids) * 2
    
    def __getitem__(self, idx):
        img_id, caption = self.img_ids[idx%len(self.img_ids)], self.captions[idx%len(self.img_ids)]
        img_id = img_id.split('_')[0]
        ffa_img = os.path.join(self.root, 'FFA', f'{img_id}_ffa.png')
        cfp_img = os.path.join(self.root, 'CFP', f'{img_id}_cfp.png')
        two_imgs = [ffa_img, cfp_img]

        img_path = random.choice(two_imgs)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, caption
    

# def split_dataset(dataset, val_ratio=0.1):
#     indices = list(range(len(dataset)))
#     train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=42)
#     return Subset(dataset, train_idx), Subset(dataset, val_idx)