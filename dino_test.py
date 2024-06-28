from transformers import Dinov2Backbone
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

# model_config = Dinov2Config.from_pretrained('./dinov2-base', image_size=(512, 512))
model = Dinov2Backbone.from_pretrained("./dinov2-small", out_indices=[-1]).to('cuda')
vits14 = torch.hub.load('./dinov2', 'dinov2_vits14', weights={'LVD142M':'./dinov2_models/dinov2_vits14_pretrain.pth'}, source='local').cuda()
# pixel_values = torch.randn(1, 3, 128, 512)

patch_h = 16
patch_w = 64
feat_dim = 384

def normalize(x):
    denominator = np.linalg.norm(x, axis=-1, keepdims=True)
    denominator = np.where(denominator == 0, 1, denominator)
    return x / denominator

def reshape_normalize(x):
    '''
    Args:
        x: [B, C, H, W]

    Returns:

    '''
    B, C, H, W = x.shape
    x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

    denominator = np.linalg.norm(x, axis=-1, keepdims=True)
    denominator = np.where(denominator==0, 1, denominator)
    return x / denominator

def single_features_to_RGB(sat_features):
    sat_feat = sat_features[:1,:,:,:].data.cpu().numpy()
    # 1. 重塑特征图形状为 [256, 64*64]
    B, C, H, W = sat_feat.shape
    flatten = np.concatenate([sat_feat], axis=0)
    # 2. 进行 PCA 降维到 3 维
    pca = PCA(n_components=3)
    pca.fit(reshape_normalize(flatten))
    
    # 3. 归一化到 [0, 1] 范围
    sat_feat_new = ((normalize(pca.transform(reshape_normalize(sat_feat))) + 1 )/ 2).reshape(B, H, W, 3)

    sat = Image.fromarray((sat_feat_new[0] * 255).astype(np.uint8))
    # sat = sat.resize((512, 512))
    sat.save('test_feat.png')

transform = T.Compose([
    T.Resize((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

img = Image.open('./grd_feat.png').convert('RGB')

imgs_tensor = transform(img).unsqueeze(0).to('cuda')
outputs = vits14.forward_features(imgs_tensor)['x_norm_patchtokens'].permute(0,2,1)

features = outputs.reshape(1, feat_dim, patch_h ,patch_w)

single_features_to_RGB(features)


