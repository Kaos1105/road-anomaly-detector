from flask import Flask
import joblib
import xgboost as xgb

def create_app():
    app = Flask(__name__)

    # Import routes after creating app to avoid circular imports
    model = xgb.XGBClassifier()
    model.load_model("model/xgb_model_binary.json")  # Load model using XGBoost's method
    app.config['model'] = model
    app.config['scaler'] = joblib.load("model/scaler_binary.pkl")
    app.config['label_encoder'] = joblib.load("model/label_encoder_binary.pkl")
    from .routes import setup_routes
    setup_routes(app)

    return app
