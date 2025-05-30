import matplotlib.pyplot as plt
import numpy as np
import torch
import cairosvg
from backend.torch.models.stable_diffusion_3_controlnet import (
    StableDiffusion3ControlNetModel,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StableDiffusion3ControlNetModel.from_repo(
    ("../stable-diffusion-3-medium-diffusers", "../SD3-Controlnet-Canny"), device
)

weights = model.controlnet.transformer_blocks[0].norm1.linear.weight
weights = weights.detach().cpu()

per_seg = 1536
reduced_out = []
for i in range(6):
    seg = weights[i * per_seg : (i + 1) * per_seg]
    seg_bins = torch.chunk(seg, 32, dim=0)
    seg_avg = [chunk.mean(dim=0) for chunk in seg_bins]
    reduced_out.append(torch.stack(seg_avg))
reduced_out = torch.cat(reduced_out, dim=0)
in_bins = torch.chunk(reduced_out, 96, dim=1)
final = torch.stack([chunk.mean(dim=1) for chunk in in_bins], dim=1)

data = final.numpy()
print(data.shape)
np.savetxt("weight.csv", data, delimiter=",")

y_vals = np.linspace(0, 9216, data.shape[0])
x_vals = np.linspace(0, 1536, data.shape[1])
X, Y = np.meshgrid(x_vals, y_vals)
Z = data
plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X, Y, Z, cmap="plasma", edgecolor=(0, 0, 0, 0.1), linewidth=0.1, antialiased=True
)
ax.set_xticklabels([])
for i in range(6):
    if i == 0:
        continue
    y_val = y_vals[i * 32]
    X_plane, Z_plane = np.meshgrid(x_vals, np.linspace(Z.min(), Z.max(), 50))
    Y_plane = np.full_like(X_plane, y_val)

    ax.plot_surface(
        X_plane, Y_plane, Z_plane, color="gray", alpha=0.2, edgecolor="none"
    )
ax.set_xlabel("Input Channel", fontsize=12, labelpad=-5)
ax.set_ylabel("Output Channel", fontsize=12, labelpad=10)
ax.set_zlabel("Weight Value", fontsize=12)
ax.set_box_aspect([1, 1, 0.3])
ax.view_init(elev=15, azim=10)
ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
ax.tick_params(axis="both", which="major", labelsize=12)
fig.savefig("weight.svg", format="svg")
cairosvg.svg2pdf(url="weight.svg", write_to="weight.pdf")
