import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Importing your custom logic modules
from face_analyzer import get_face_score
from voice import get_voice_score
from model import calculate_risk

app = Flask(__name__)
CORS(app)

# Create a temporary directory to save uploaded assets for processing
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/analyze", methods=["POST"])
def analyze():
    # 1. Validation: Ensure both required files are included in the request
    if "image" not in request.files or "audio" not in request.files:
        return jsonify({
            "error": "Missing files. Both 'image' and 'audio' parts are required."
        }), 400

    image_file = request.files["image"]
    audio_file = request.files["audio"]

    # 2. Validation: Ensure the user actually selected files
    if image_file.filename == "" or audio_file.filename == "":
        return jsonify({"error": "Empty files submitted."}), 400

    try:
        # 3. Secure filenames and save files temporarily to disk
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(image_file.filename))
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(audio_file.filename))
        
        image_file.save(img_path)
        audio_file.save(audio_path)

        # 4. Run the files through your media analytical engines
        face_score = get_face_score(img_path)
        voice_score = get_voice_score(audio_path)

        # 5. Compute the risk metrics based on calculated scores
        result = calculate_risk(face_score, voice_score)

        # 6. Clean up the files from the disk to prevent server bloat
        if os.path.exists(img_path): 
            os.remove(img_path)
        if os.path.exists(audio_path): 
            os.remove(audio_path)

        # Add raw metrics into the final JSON package for frontend visibility
        result["face_score"] = round(face_score, 2)
        result["voice_score"] = round(voice_score, 2)
        
        return jsonify(result), 200

    except Exception as e:
        # Fallback guard to prevent the server from completely crashing on bad data
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
