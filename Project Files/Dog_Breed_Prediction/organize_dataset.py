import os
import shutil
import pandas as pd



dataset_dir = r"C:\Dog_Breed_Prediction\dataset\train"
labels = pd.read_csv(r"C:\Dog_Breed_Prediction\dataset\labels.csv")



def make_dir(x):
    if not os.path.exists(x):
        os.makedirs(x)

base_dir = r"C:\Dog_Breed_Prediction\subset"
make_dir(base_dir)

train_dir = os.path.join(base_dir, 'train')
make_dir(train_dir)



breeds = labels.breed.unique()

for breed in breeds:
    
    breed_folder = os.path.join(train_dir, breed)
    make_dir(breed_folder)

    images = labels[labels.breed == breed]['id']

    for image in images:
        source = os.path.join(dataset_dir, f"{image}.jpg")
        destination = os.path.join(breed_folder, f"{image}.jpg")

        if os.path.exists(source):
            shutil.copyfile(source, destination)

print("Dataset organized successfully!")
