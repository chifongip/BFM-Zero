import h5py

file_path = "/home/ubuntu/BFM-Zero/new_model_for_training_code_inference/checkpoint/buffers/train/buffer.hdf5"
with h5py.File(file_path, "r") as f:
    keys = list(f.keys())
    for key in keys:
        dataset = f[key]
        print(f"{key}: shape={dataset.shape}, dtype={dataset.dtype}")