import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from pathlib import Path

# --- Parameters ---
image_folder = Path("/user/jpucek/BSRH/ForDaniele/")  # <-- CHANGE THIS
max_images_per_group = 5
scaling_factors = [15.0, 1.0, 17.0, 2.5, 1.0]  # <-- One per group (extend as needed)
center_offset_h = [0.012, 0, -0.0072, 0, 0.016, 0]
center_offset_v = [0, 0, 0, 0, 0, 0]

selected_group_indices = [0, 2, 5, 4]  # e.g., plot groups 1 and 3 only
custom_labels = ["LS opened, no occulter", "LS closed, no occulter", "LS opened, occulted", "LS closed, occulted"]

# --- Load images and compute projections ---
png_files = sorted(image_folder.glob("*.png"))
x_hor = []
x_ver = []
horizontal_projections = []
vertical_projections = []
image_names = []

for image_path in png_files:
    print(f"Reading: {image_path.name}")
    img = iio.imread(image_path, plugin="pillow")
    if img.dtype != np.uint16:
        print(f"Skipping {image_path.name} (not 16-bit).")
        continue

    horizontal_projections.append(np.sum(img[150:450,:], axis=0))
    vertical_projections.append(np.sum(img[:,330:630], axis=1))
    image_names.append(image_path.stem)

# --- Group images by order into chunks of ≤5 ---
final_groups = []
final_group_names = []

for i in range(0, len(image_names), max_images_per_group):
    chunk_indices = list(range(i, min(i + max_images_per_group, len(image_names))))
    final_groups.append(chunk_indices)
    name_list = [image_names[j] for j in chunk_indices]
    final_group_names.append(f"Group {len(final_groups)} ({len(chunk_indices)})")

# Pad scaling_factors if needed
if len(scaling_factors) < len(final_groups):
    scaling_factors.extend([1.0] * (len(final_groups) - len(scaling_factors)))

# --- Compute average projections with BG subtraction and scaling ---
avg_horizontal = []
avg_vertical = []

# for group_indices, scale in zip(final_groups, scaling_factors):
#     h_stack = np.stack([horizontal_projections[i] for i in group_indices])
#     v_stack = np.stack([vertical_projections[i] for i in group_indices])
#     h_avg = np.mean(h_stack, axis=0)
#     v_avg = np.mean(v_stack, axis=0)
group = 0
for group_indices, scale in zip(final_groups, scaling_factors):
    i = group_indices[3]  # first image index in the group
    h_avg = np.asarray(horizontal_projections[i].copy())
    v_avg = np.asarray(vertical_projections[i].copy())

    # Background subtraction: subtract mean of first 50 pixels
    h_avg = h_avg - np.mean(h_avg[:50])
    v_avg = v_avg - np.mean(v_avg[:50])

    # Apply scaling
    h_avg *= scale
    v_avg *= scale

    x_hor.append(np.linspace(-1.0648+center_offset_h[group], 1.0648+center_offset_h[group], len(h_avg)))
    avg_horizontal.append(h_avg)
    x_ver.append(np.linspace(-0.5363 + center_offset_v[group], 0.5363 + center_offset_v[group], len(v_avg)))
    avg_vertical.append(v_avg)
    group += 1
# --- Final Plot ---
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Horizontal subplot
for idx, label in zip(selected_group_indices, custom_labels):
    axs[0].plot(x_hor[idx], avg_horizontal[idx], label=label)
    axs[1].plot(x_ver[idx], avg_vertical[idx], label=label)
axs[0].set_title("Avg Horizontal Projections (Ordered Groups)")
axs[0].set_xlabel("X (columns)")
axs[0].set_ylabel("Intensity")
axs[0].legend(fontsize='x-small')

# Vertical subplot
axs[1].set_title("Avg Vertical Projections (Ordered Groups)")
axs[1].set_xlabel("Y (rows)")
axs[1].set_ylabel("Intensity")
axs[1].legend(fontsize='x-small')

plt.suptitle("Ordered Grouped Projections with BG Subtraction and Scaling")
plt.tight_layout()
plt.show()
