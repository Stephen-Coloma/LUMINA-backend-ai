import torch
import joblib

# AI Model
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ai_model = torch.jit.load("C:/Users/Renuel Balogo/Documents/Projects Cloned/nsclc-classifier-ai/agent/app/models/best_model.pt")
ai_model = ai_model.to(gpu)
ai_model.eval()

# Machine Learning Model
data_sci_model= joblib.load("C:/Users/Renuel Balogo/Documents/Projects Cloned/nsclc-classifier-ai/agent/app/models/logistic_model.pkl")