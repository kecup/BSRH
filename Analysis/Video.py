import dill as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import pyjapc
import zstandard as zstd

# Setup
dctx = zstd.ZstdDecompressor()
directory = '/user/jpucek/BSRH/Data/2025/06/19/'
files = sorted(os.listdir(directory))
japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')

bunch = 160
file_range = range(109, 162)

# Frame map: global_frame_index -> (file_idx, local_idx_within_file_cache)
frame_map = []

# Mapping from global file_idx to actual filename
file_idx_map = {}

# Build index of valid frames, but don’t load images yet
for local_file_idx, file_idx in enumerate(file_range):
    file_path = os.path.join(directory, files[file_idx])
    file_idx_map[file_idx] = file_path
    with open(file_path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            try:
                a = pickle.load(reader)
                local_valid_idx = 0
                for frame in a:
                    gate = frame.device_data['GATE-INT']
                    if gate['delaySlots'] == bunch and gate['pulseLength'] >= 400:
                        frame_map.append((file_idx, local_valid_idx))
                        local_valid_idx += 1
            except Exception as e:
                print(f"Error reading {files[file_idx]}: {e}")

print(f"Total valid frames: {len(frame_map)}")

# Runtime cache
cached_file_idx = None
cached_images = []

def load_and_cache_file(file_idx):
    """Load matching images from one file and cache them."""
    global cached_file_idx, cached_images
    cached_images = []
    file_path = file_idx_map[file_idx]
    with open(file_path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            a = pickle.load(reader)
            for frame in a:
                gate = frame.device_data['GATE-INT']
                if gate['delaySlots'] == bunch and gate['pulseLength'] >= 400:
                    image = frame.cam_data['image2D']
                    image = image - np.mean(image[:100, :100])
                    cached_images.append(image)
    cached_file_idx = file_idx

# Setup animation figure
fig, ax = plt.subplots()
tst = ax.imshow(np.zeros((600, 800)), cmap='jet', vmin=0, vmax=3000)
plt.colorbar(tst, ax=ax)

def update(frame_number):
    global cached_file_idx
    file_idx, local_idx = frame_map[frame_number]

    if cached_file_idx != file_idx:
        load_and_cache_file(file_idx)

    image = cached_images[local_idx]
    tst.set_data(image)
    return (tst,)

ani = animation.FuncAnimation(
    fig=fig,
    func=update,
    frames=len(frame_map),
    interval=100,
    repeat=False
)

writer = animation.PillowWriter(fps=20, metadata=dict(artist='Me'))
ani.save(f'Video_{bunch}.gif', writer=writer)

# plt.show()
