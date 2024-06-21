import numpy as np
import torch
import warnings
import gradio as gr
from torch_geometry import get_perspective_transform, warp_perspective

warnings.filterwarnings("ignore")

Fov = 17.5
Pitch = 0.8
Scale = 10

def get_BEV_kitti(front_imgs, out_size, heading, Tx, Ty):
    device = heading.device
    gamma = heading.unsqueeze(-1) + torch.tensor(90 * torch.pi / 180).to(device)
    
    B, S, C, Hp, Wp = front_imgs.shape

    Wo, Ho = torch.tensor(float(Wp * Scale)), torch.tensor(float(Wp * Scale))

    fov = torch.tensor(Fov * torch.pi / 180)
    theta = torch.tensor(Pitch * torch.pi / 180)

    f = Hp / 2 / torch.tan(torch.tensor(fov))
    phi = torch.pi / 2 - fov
    delta = torch.pi / 2 + theta - torch.tensor(phi)
    l = torch.sqrt(f**2 + (Hp / 2)**2)
    h = l * torch.sin(delta)
    f_ = l * torch.cos(delta)

    frame = front_imgs.clone()
    frame[:, :, :, :Hp // 2, :] = 0
    
    y = (torch.ones((2, 2)).to(device).T  *(torch.arange(0,Ho, step=Ho-1)).to(device)).T
    x = torch.ones((2, 2)).to(device) * torch.arange(0, Wo, step=Wo-1).to(device)
    l0 = torch.ones((2, 2)).to(device)*Ho - y
    l1 = torch.ones((2, 2)).to(device) * f_+ l0
    
    f1_0 = torch.arctan(h / l1)
    f1_1 = torch.ones((2, 2)).to(device)*(torch.pi/2+theta) - f1_0
    y_ = l0 * torch.sin(f1_0) / torch.sin(f1_1)
    j_p = torch.ones((2, 2)).to(device) * Hp - y_
    i_p = torch.ones((2, 2)).to(device) * Wp/2 -(f_+torch.sin(torch.tensor(theta))*(torch.ones((2, 2)).to(device)*Hp-j_p))*(Wo/2*torch.ones((2, 2)).to(device)-x)/l1
    
    i_p = i_p.unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)
    j_p = j_p.unsqueeze(0).unsqueeze(0).repeat(B, S, 1, 1)
    
    four_point_org = torch.stack((i_p, j_p), dim=-1).view(B, S, 4, 2)
    four_point_new = torch.stack((x, y), dim=-1).view(1, 1, 4, 2).repeat(B, S, 1, 1)
    
    H = get_perspective_transform(four_point_org.view(-1, 4, 2), four_point_new.view(-1, 4, 2)).view(B, S, 3, 3)
    
    scale1, scale2 = out_size / Wo, out_size / Ho
    
    H1 = torch.tensor([
        [scale1, 0, 0],
        [0, scale2, 0],
        [0, 0, 1]
    ], dtype=torch.float32).to(device)
    
    uc = float(out_size / 2)
    vc = float(out_size / 2)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)
    H2 = torch.stack([
        torch.cat([cos_gamma, -sin_gamma, uc * (1 - cos_gamma) + vc * sin_gamma], dim=-1),
        torch.cat([sin_gamma, cos_gamma, uc * (1 - cos_gamma) - vc * sin_gamma], dim=-1),
        torch.tensor([0, 0, 1], dtype=torch.float32, device=device).expand(B, S, -1)
    ], dim=-2)
    
    T_translate_back = torch.stack([
        torch.cat([torch.ones_like(Tx), torch.zeros_like(Tx), (uc * sin_gamma) + Tx], dim=-1),
        torch.cat([torch.zeros_like(Tx), torch.ones_like(Tx), (-vc * cos_gamma) + Ty], dim=-1),
        torch.tensor([0, 0, 1], dtype=torch.float32, device=device).expand(B, S, -1)
    ], dim=-2)
    
    Homo = torch.matmul(T_translate_back, torch.matmul(H2, H1))
    Homo = torch.matmul(Homo, H.view(B, S, 3, 3))
    
    frames_reshaped = frame.view(B * S, C, Hp, Wp).float()
    Homo_reshaped = Homo.view(B * S, 3, 3)
    BEVs = warp_perspective(frames_reshaped, Homo_reshaped, (out_size, out_size)).view(B, S, C, out_size, out_size)
    
    return BEVs

@torch.no_grad()
def KittiBEV():
    torch.cuda.empty_cache()
    
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # HC-Net: Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator
            ## Get BEV from front-view image. 
            [[Paper](https://arxiv.org/abs/2308.16906)]  [[Code](https://github.com/xlwangDev/HC-Net)]
            """)

        with gr.Row():
            front_img = gr.Image(label="Front-view Image").style(height=450)
            BEV_output = gr.Image(label="BEV Image").style(height=450)

        fov = gr.Slider(1,90, value=20, label="FOV")
        pitch = gr.Slider(-180, 180, value=0, label="Pitch")
        scale = gr.Slider(1, 10, value=1.0, label="Scale")
        out_size = gr.Slider(500, 1000, value=500, label="Out size")
        btn = gr.Button(value="Get BEV Image")
        btn.click(get_BEV_kitti,inputs= [front_img, fov, pitch, scale, out_size], outputs=BEV_output, queue=False)
        gr.Markdown(
            """
            ### Note: 
            - 'FOV' represents the field of view in the camera's vertical direction, please refer to section A.2 in the [paper](https://arxiv.org/abs/2308.16906)'s Supplementary.
            - By default, the camera faces straight ahead, with a 'pitch' of 0 resulting in a top-down view. Increasing the 'pitch' tilts the BEV view upwards.
            - 'Scale' affects the field of view in the BEV image; a larger 'Scale' includes more content in the BEV image.
            """
        )


        gr.Markdown("## Image Examples")
        gr.Examples(
            examples=[['./figure/exp1.jpg', 27, 7, 6, 1000],
                      ['./figure/exp2.png', 17.5, 0.8, 4, 1000],
                      ['./figure/exp3.jepg', 48, 11.5, 7, 1000]],
            inputs= [front_img, fov, pitch, scale, out_size],
            outputs=[BEV_output],
            fn=get_BEV_kitti,
            cache_examples=False,
        )
    demo.launch()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    KittiBEV()