import torch

def load_model():
    device = torch.device("cpu")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    model.to(device)
    model.eval()
    return model
