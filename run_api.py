from src.app.api import create_app

# Tell create_app where the trained model is
app = create_app(model_path="models/breast_cancer_model.pkl")