import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = load_model("dogbreed.h5")

class_names = [
    'affenpinscher','beagle','appenzeller','basset','bluetick',
    'boxer','cairn','doberman','german_shepherd','golden_retriever',
    'kelpie','komondor','leonberg','mexican_hairless','pug',
    'redbone','shih-tzu','toy_poodle','vizsla','whippet'
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = load_img(filepath, target_size=(224, 224))
        x = img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)
        predicted_class = class_names[np.argmax(preds)]
       

        return render_template("output.html",
                               prediction=predicted_class)

    return render_template("predict.html")

if __name__ == '__main__':
    app.run(debug=True)
