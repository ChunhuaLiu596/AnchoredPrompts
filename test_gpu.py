import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))


# True
# 1
# 0
# NVIDIA A100-PCIE-40GB
