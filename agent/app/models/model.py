import torch
import joblib
import os

# AI Model
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ai_model_path = os.path.join("app", "models", "best_model.pt")
ai_model = torch.jit.load(ai_model_path)
ai_model = ai_model.to(gpu)
ai_model.eval()

# Machine Learning Model
ml_model_path = os.path.join("app", "models", "logistic_model.pkl")
data_sci_model = joblib.load(ml_model_path)
