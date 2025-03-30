from flask import jsonify, request
from .preprocess import preprocess_api, spectrogram_img

def setup_routes(app):
    @app.route('/')
    def home():
        return jsonify({"message": "Python Flask Server Running on Android!"})
    
    @app.route('/preprocess')
    def preprocess():
        return preprocess_api(request)
        
    @app.route('/spectrogram', methods=['POST'])
    def spectrogram_api():
        return spectrogram_img(request);