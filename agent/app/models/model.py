import torch

ai_model = torch.load("temp_model.pth")
ai_model.eval()

data_sci_model= torch.load("model_b.pth")
data_sci_model.eval()