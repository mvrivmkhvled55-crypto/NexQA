import os
import cv2
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
from pathlib import Path

# Configuration matching model.py
CONFIG = {
    "model_path": "yolov8n.pt",  # Pretrained fallback
    "best_weights": "best.pt",
    "class_names_ar": [
        "كوارتزيت (Quartzity)",
        "عقدة حية (Live Knot)",
        "نخاع (Marrow)",
        "راتنج (Resin)",
        "عقدة ميتة (Dead Knot)",
        "عقدة مشققة (Knot with Crack)",
        "عقدة مفقودة (Knot missing)",
        "شرخ (Crack)",
    ],
}

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

@app.before_request
def log_request_info():
    print(f"📥 Request: {request.method} {request.path} from {request.remote_addr}")

# Database Configuration
# Railway provides DATABASE_URL environment variable
db_url = os.getenv('DATABASE_URL', 'sqlite:///wood_defects.db')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class UserRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    factory_name = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class ProductRecord(db.Model):
    id = db.Column(db.String(50), primary_key=True)
    image_base64 = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    defect_type = db.Column(db.String(100))
    buyer = db.Column(db.String(100))
    shipping = db.Column(db.String(100))
    inspected_at = db.Column(db.String(50))

with app.app_context():
    db.create_all()

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model
if os.path.exists(CONFIG["best_weights"]):
    model = YOLO(CONFIG["best_weights"])
    print(f"✅ Loaded best weights from {CONFIG['best_weights']}")
else:
    model = YOLO(CONFIG["model_path"])
    print(f"⚠️  Best weights not found. Using pretrained {CONFIG['model_path']}")

def process_image(img_path):
    img = cv2.imread(str(img_path))
    results = model.predict(img, conf=0.25)
    
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            name = CONFIG["class_names_ar"][cls_id] if cls_id < len(CONFIG["class_names_ar"]) else f"Unknown ({cls_id})"
                
            detections.append({
                "name": name,
                "confidence": f"{conf:.1%}",
                "bbox": xyxy
            })
            
            # Draw for display
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    processed_path = Path(UPLOAD_FOLDER) / f"proc_{img_path.name}"
    cv2.imwrite(str(processed_path), img)
    return detections, processed_path.name

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NexQA - Model Test</title>
        <style>
            body { font-family: 'Inter', sans-serif; background: #0f1117; color: white; text-align: center; padding: 50px; }
            .container { max-width: 600px; margin: 0 auto; background: #1a1d27; padding: 30px; border-radius: 20px; border: 1px solid #3a3f55; }
            h1 { color: #00E5FF; margin-bottom: 30px; }
            input[type="file"] { margin: 20px 0; }
            button { background: #00E5FF; color: black; border: none; padding: 12px 25px; border-radius: 10px; font-weight: bold; cursor: pointer; }
            #result { margin-top: 30px; }
            img { max-width: 100%; border-radius: 10px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>NexQA Model Tester</h1>
            <p>Upload a wood image to test detections</p>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required><br>
                <button type="submit">Analyze Image</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    img_path = Path(UPLOAD_FOLDER) / file.filename
    file.save(img_path)
    detections, proc_name = process_image(img_path)
    
    # Return HTML for browser testing
    detections_html = "".join([f"<li><b>{d['name']}</b>: {d['confidence']}</li>" for d in detections])
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>NexQA Result</title>
        <style>
            body {{ font-family: sans-serif; background: #0f1117; color: white; text-align: center; padding: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; background: #1a1d27; padding: 20px; border-radius: 20px; }}
            img {{ max-width: 100%; border-radius: 10px; border: 2px solid #00E5FF; }}
            ul {{ list-style: none; padding: 0; }}
            li {{ background: #2a2d37; margin: 10px; padding: 10px; border-radius: 8px; }}
            a {{ color: #00E5FF; text-decoration: none; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Analysis Result</h1>
            <img src="/static/uploads/{proc_name}" alt="Processed Image">
            <h3>Detections:</h3>
            <ul>{detections_html if detections_html else "<li>No defects detected</li>"}</ul>
            <br>
            <a href="/">← Test Another Image</a>
        </div>
    </body>
    </html>
    """

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Endpoint for Mobile App (Flutter)"""
    if 'image' not in request.files:
        print("❌ No image provided in request")
        return jsonify({"success": False, "message": "No image provided"}), 400
    
    file = request.files['image']
    img_path = Path(UPLOAD_FOLDER) / f"mobile_{file.filename}"
    file.save(img_path)
    print(f"📸 Received image: {file.filename}")
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Failed to read image: {img_path}")
        return jsonify({"success": False, "message": "Invalid image data"}), 400

    results = model.predict(img, conf=0.25, verbose=False)
    
    final_results = []
    for result in results:
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                final_results.append({
                    "class_id": cls_id,
                    "name": CONFIG["class_names_ar"][cls_id],
                    "confidence": round(conf, 4),
                    "box": {"x1": round(xyxy[0], 2), "y1": round(xyxy[1], 2), "x2": round(xyxy[2], 2), "y2": round(xyxy[3], 2)}
                })
    
    print(f"🔍 Detections: {len(final_results)}")
    
    # Save to Database
    status = 'rejected' if len(final_results) > 0 else 'passed'
    defect_name = final_results[0]['name'] if len(final_results) > 0 else 'None'
    conf = final_results[0]['confidence'] if len(final_results) > 0 else 100.0
    
    # Read image as base64 for database storage
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    import datetime
    record_id = f"#{db.session.query(ProductRecord).count() + 1}"
    new_record = ProductRecord(
        id=record_id,
        image_base64=encoded_string,
        status=status,
        confidence=conf,
        defect_type=defect_name,
        buyer="Global Tech", # Placeholder
        shipping="DHL",      # Placeholder
        inspected_at=datetime.datetime.now().isoformat()
    )
    db.session.add(new_record)
    db.session.commit()

    return jsonify({
        "success": True, 
        "data": final_results,
        "record_id": record_id
    })

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    print(f"🔥 Internal Error: {e}")
    return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/signup', methods=['POST'])
def api_signup():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"success": False, "message": "No JSON data received"}), 400
            
        factory_name = data.get('factory_name')
        password = data.get('password')
        
        print(f"📝 Attempting signup for: {factory_name}")
        
        if UserRecord.query.filter_by(factory_name=factory_name).first():
            return jsonify({"success": False, "message": "Factory already exists"}), 400
        
        new_user = UserRecord(factory_name=factory_name, password=password)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"success": True, "message": "User created successfully"})
    except Exception as e:
        print(f"❌ Signup Error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    factory_name = data.get('factory_name')
    password = data.get('password')
    
    user = UserRecord.query.filter_by(factory_name=factory_name, password=password).first()
    if user:
        return jsonify({"success": True, "factory_name": user.factory_name})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
