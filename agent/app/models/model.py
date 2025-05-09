import torch
from app.models.ai.nsclc_model import NSCLC_Model

# ai_model = NSCLC_Model(model_config=None)  # Assuming model_config is defined somewhere in your code
# ai_model.load_state_dict(torch.load("C:/Users/Renuel Balogo/Documents/Projects Cloned/nsclc-classifier-ai/agent/app/models/best_model.pth", weights_only=True, map_location=torch.device('cpu')))
# # ai_model.eval()
ai_model = torch.jit.load("C:/Users/Renuel Balogo/Documents/Projects Cloned/nsclc-classifier-ai/agent/app/models/best_model.pt")
ai_model.eval()
# data_sci_model= torch.load("model_b.pth")
# data_sci_model.eval()