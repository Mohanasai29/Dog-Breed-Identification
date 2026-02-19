import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import load_model

model = load_model("dogbreed.h5")

img_path = "dataset/train/0dc45e3e57bbcfccc550479d57b39951.jpg"


img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = preprocess_input(x)
x = np.expand_dims(x, axis=0)


preds = model.predict(x)


class_names = [
    'affenpinscher','beagle','appenzeller','basset','bluetick',
    'boxer','cairn','doberman','german_shepherd','golden_retriever',
    'kelpie','komondor','leonberg','mexican_hairless','pug',
    'redbone','shih-tzu','toy_poodle','vizsla','whippet'
]

predicted_class = class_names[np.argmax(preds)]


print("Predicted Breed:", predicted_class)

