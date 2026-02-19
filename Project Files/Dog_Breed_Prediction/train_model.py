import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

selected_classes = [
    'affenpinscher','beagle','appenzeller','basset','bluetick',
    'boxer','cairn','doberman','german_shepherd','golden_retriever',
    'kelpie','komondor','leonberg','mexican_hairless','pug',
    'redbone','shih-tzu','toy_poodle','vizsla','whippet'
]

datagen = ImageDataGenerator()

generator = datagen.flow_from_directory(
    directory=r"C:\Dog_Breed_Prediction\subset\train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    classes=selected_classes
)

print("Data loaded successfully!")


Image_size = [224,224]

sol = VGG19(
    input_shape = Image_size + [3], 
    weights = 'imagenet', 
    include_top = False
)

for i in sol.layers:
    i.trainable = False

y = Flatten()(sol.output)

final = Dense(20, activation='softmax')(y)

vgg19_model = Model(inputs=sol.input, outputs=final)

vgg19_model.summary()

vgg19_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']   
)


vgg19_model.fit(generator, epochs=6)

vgg19_model.save("dogbreed.h5")
