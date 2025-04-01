from flask import jsonify, request
# from .preprocess import preprocess_api, spectrogram_img
from .ml_process import predict

def setup_routes(app):
    @app.route('/', methods=['GET'])
    def home():
        return jsonify({"message": "Python Flask Server Running on Android!"})
    
    # @app.route('/preprocess')
    # def preprocess():
    #     return preprocess_api(request)
    
    @app.route('/predict', methods=['POST'])
    def predict_api():
        return predict(request, app)
        
    # @app.route('/spectrogram', methods=['POST'])
    # def spectrogram_api():
    #     return spectrogram_img(request);