import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Generator Model (same as training)
# -------------------------
class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Load Model
# -------------------------
z_dim = 100
device = torch.device("cpu")

gen = Generator(z_dim).to(device)
gen.load_state_dict(torch.load("/Users/jindsaini/Desktop/Lab5/best_generator_kaggle.pth", map_location=device))
gen.eval()

# -------------------------
# Streamlit UI
# -------------------------
st.title("🎨 WGAN CIFAR Image Generator")

st.write("Generate new CIFAR-like images using trained WGAN")

num_images = st.slider("Number of images", 1, 25, 9)

if st.button("Generate Images"):
    noise = torch.randn(num_images, z_dim, 1, 1).to(device)
    fake_images = gen(noise).detach().cpu()

    grid = vutils.make_grid(fake_images, nrow=int(np.sqrt(num_images)), normalize=True)
    
    plt.figure(figsize=(5,5))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.axis("off")

    st.pyplot(plt)