import torch
print(torch.version.cuda)  # CUDA version used by PyTorch
print(torch.backends.cudnn.version())  # cuDNN version
print(torch.cuda.is_available())  # Check if CUDA is accessible

