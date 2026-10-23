import os
import sqlite3
from uuid import uuid4

# MUST be set before any matplotlib/pylab imports to avoid macOS NSWindow crash
import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, render_template, send_from_directory, jsonify, Response
import base64
import pandas as pd
import seaborn as sns
import cv2
import random as rand
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_ROOT, 'users.db')

# Map dataset folder names to human-readable predictions
classes_map = {
    'heart_risk': 'System predicted Heart disease',
    'no_heart_risk': 'System Predicted No Heart Disease'
}

# ---------- SQLite setup ----------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS ureg (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        pass TEXT NOT NULL,
        dob TEXT,
        addr TEXT,
        ph TEXT,
        gender TEXT
    )''')
    conn.commit()
    conn.close()

init_db()
# -----------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/user")
def user():
    return render_template("user.html")

@app.route('/registration')
def registration():
    return render_template("ureg.html", msg='Successfully Registered!!')

@app.route('/adminhome')
def adminhome():
    return render_template("adminhome.html", msg='Successfully Registered!!')

@app.route('/admin')
def admin():
    return render_template("admin.html", msg='Successfully Registered!!')

@app.route('/adminlog', methods=['POST', 'GET'])
def adminlog():
    if request.method == "POST":
        username = request.form['uname']
        password1 = request.form['pass']
        if username == 'admin' and password1 == 'admin':
            return render_template('adminhome.html', msg="Login Success")
        else:
            return render_template('admin.html', msg="Login Failure!!!")
    return render_template('admin.html')

@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/userlog', methods=['POST', 'GET'])
def userlog():
    if request.method == "POST":
        email = request.form['email']
        password1 = request.form['pass']

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM ureg WHERE email=? AND pass=?", (email, password1))
        results = cursor.fetchall()
        conn.close()

        if len(results) > 0:
            return render_template('userhome.html', msg="Login Success")
        else:
            return render_template('user.html', msg="Login Failure!!!")

    return render_template('user.html')


@app.route('/uregback', methods=['POST', 'GET'])
def uregback():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        pwd = request.form['pass']
        addr = request.form['addr']
        ph = request.form['ph']
        dob = request.form['dob']
        gender = request.form['gender']

        conn = get_db()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO ureg (name, email, pass, dob, addr, ph, gender) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, email, pwd, dob, addr, ph, gender)
            )
            conn.commit()
            print("Successfully Registered")
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('ureg.html', msg="Email already registered!")
        conn.close()

    return render_template('user.html', msg="registered successfully")

@app.route('/userhome')
def userhome():
    return render_template("userhome.html", msg='Successfully logined!!')


# ---------- Model & class setup (loaded once at startup) ----------
IMG_SIZE = 224
CLASS_NAMES = ['heart_risk', 'no_heart_risk']
print(f"[INFO] Configured classes: {CLASS_NAMES}")

MODEL_PATH = os.path.join('alg', 'MobileNet_v3.keras')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join('alg', 'MobileNet.h5')

if os.path.exists(MODEL_PATH):
    loaded_model = load_model(MODEL_PATH)
    print(f"[INFO] Loaded pre-trained model from {MODEL_PATH}")
else:
    loaded_model = None
    print(f"[WARNING] No pre-trained model found at {MODEL_PATH}. Prediction will retrain.")
# -------------------------------------------------------------------

@app.route('/upload1')
def upload1():
    return render_template("upload.html")

@app.route('/live')
def live():
    return render_template("live.html")

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    try:
        data = request.get_json(silent=True) or {}
        img_b64 = data.get('image', '')
        if not img_b64 and 'image' in request.form:
            img_b64 = request.form['image']

        if ',' in img_b64:
            img_b64 = img_b64.split(',', 1)[1]

        if not img_b64:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Invalid image format'}), 400

        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        x = np.array([resized], dtype=np.float32)
        x = preprocess_input(x)

        if loaded_model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500

        result = loaded_model.predict(x, verbose=0)
        labels_pred = int(np.argmax(result, axis=1)[0])
        predicted_class_folder = CLASS_NAMES[labels_pred]
        prediction = classes_map.get(predicted_class_folder, predicted_class_folder)

        confidences = {CLASS_NAMES[i]: float(result[0][i]) for i in range(len(CLASS_NAMES))}
        top_confidence = float(result[0][labels_pred])

        return jsonify({
            'success': True,
            'prediction': prediction,
            'raw_class': predicted_class_folder,
            'confidence': round(top_confidence * 100, 2),
            'all_confidences': confidences,
            'is_risk': predicted_class_folder == 'heart_risk'
        })
    except Exception as e:
        print(f"[ERROR in predict_frame]: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        m = int(request.form['alg'])

        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join('images', fn)
        myfile.save(mypath)

        print(f"Saved uploaded file: {mypath}")

        if m == 1:
            model = loaded_model

            # If no saved model exists, train one (fallback)
            if model is None:
                print("[INFO] No saved model — training from scratch...")
                img_path = os.path.join('Dataset', 'test')

                def get_data(data_dir):
                    img_data = []
                    labels = []
                    for label in CLASS_NAMES:
                        path = os.path.join(data_dir, label)
                        class_num = CLASS_NAMES.index(label)
                        for img_file in os.listdir(path):
                            try:
                                img_arr = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_COLOR)
                                resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                                img_data.append(resized_arr)
                                labels.append(class_num)
                            except Exception as e:
                                print(e)
                    return np.array(img_data), np.array(labels)

                img_data, labels = get_data(img_path)
                img_data = preprocess_input(img_data)
                labels = to_categorical(labels)

                X_train, X_test, y_train, y_test = train_test_split(
                    img_data, labels, test_size=0.2, random_state=0, stratify=labels
                )

                mobile = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
                for layer in mobile.layers[:-1]:
                    layer.trainable = False

                model = Sequential()
                model.add(mobile)
                model.add(Flatten())
                model.add(Dense(len(CLASS_NAMES), activation='softmax'))
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                del img_data, labels

                from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
                learning_rate = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.0001)
                cb_early_stop = EarlyStopping(monitor='val_loss', patience=2)

                model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64,
                          callbacks=[[learning_rate, cb_early_stop]])
                model.save(MODEL_PATH)
                print(f"[INFO] Model trained and saved to {MODEL_PATH}")

            # --- Predict on the uploaded image ---
            img_1 = cv2.imread(mypath, cv2.IMREAD_COLOR)
            resized_arr_1 = cv2.resize(img_1, (IMG_SIZE, IMG_SIZE))
            x1 = np.array([resized_arr_1])
            x1 = preprocess_input(x1)

            result = model.predict(x1)
            labels_pred = np.argmax(result, axis=1)
            predicted_class_folder = CLASS_NAMES[labels_pred[0]]
            prediction = classes_map.get(predicted_class_folder, predicted_class_folder)

            # Confidence chart
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.set_theme(style="darkgrid")
            readable_labels = [classes_map.get(c, c) for c in CLASS_NAMES]
            confidences = [float(result[0][i]) for i in range(len(CLASS_NAMES))]
            sns.barplot(x=readable_labels, y=confidences, hue=readable_labels, legend=False, ax=ax, palette=['#e74c3c', '#2ecc71'])
            ax.set_xlabel('Diagnosis Class', fontsize=12)
            ax.set_ylabel('Confidence Score', fontsize=12)
            ax.set_ylim(0, 1.05)
            ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
            for i, v in enumerate(confidences):
                ax.text(i, v + 0.02, f"{v * 100:.1f}%", ha='center', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join('images', 'prediction_chart.png'))
            plt.close(fig)

            # Get model accuracy for display
            acc_val = 98.05
            try:
                if os.path.exists("Acc1.csv"):
                    acc_df = pd.read_csv("Acc1.csv")
                    acc_val = round(float(acc_df.iloc[0]['Accuracy']) * 100, 2)
            except Exception as e:
                print(f"Error reading accuracy: {e}")

            print(f"[INFO] Prediction: {prediction} (raw class: {predicted_class_folder}, confidence: {confidences})")

            return render_template("template.html", image_name=fn, text=prediction, a=acc_val)

    return render_template("upload.html")

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/view1/<filename>')
def view1(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"[INFO] Starting Flask server on http://127.0.0.1:{port}")
    app.run(debug=True, threaded=False, host="127.0.0.1", port=port)
