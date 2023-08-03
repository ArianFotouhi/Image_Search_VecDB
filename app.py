import pinecone
import requests
import os
import zipfile

pinecone.init(api_key="0b5323aa-73bc-4730-b9bd-cbe225ab9849", environment="gcp-starter")

DATA_DIR = "temp"
IMAGENET_DIR = f"{DATA_DIR}/tiny-imagenet-200"
IMAGENET_ZIP = f"{DATA_DIR}/tiny-imagenet-200.zip"
IMAGENET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"

def download_data():
    os.mkdir(DATA_DIR, exist_ok=True)

    if not os.path.exists(IMAGENET_DIR):
        if not os.path.exists(IMAGENET_ZIP):
            r = requests.get(IMAGENET_URL)
            with open(IMAGENET_ZIP, 'wb') as f:
                f.write(r.content)
        with zipfile.ZipFile(IMAGENET_ZIP, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

download_data()

from torchvision import datasets
import random

image_classes = set(random.simple(range(200), 5) )

image_file_names = [
    file_name
    for file_name, label in datasets.ImageFolder(f"{IMAGENET_DIR}/train").imgs
    if label in image_classes
]

import matplotlib.pyplot as plt
from PIL import Image

def show_images_horizentally(file_names):
    m = len(file_names)
    fig, ax = plt.subplots(1,m)
    fig.set_figwidth(1.5*m)
    for a, f in zip(ax, file_names):
        a.imshow(Image.open(f))
        a.axis('off')
    plt.show()

def show_image(file_name):
    fig, ax = plt.subplots(1,1)
    fig.set_figwidth(1.3)
    ax.imshow(Image.open(file_name))
    ax.axis('off')


#conver images to embeddings
from torchvision import transforms as ts
import torchvision.models as models

class ImageEmbedder:
    def __init__(self):
        self.normalize = ts.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
        self.model = models.squeezenet1_0(pretrained = True,
                                          progress = False)
    
    def embed(self, image_file_name):
        image = Image.open(image_file_name).convert('RGB')
        image = ts.Resize(256)(image)
        image = ts.CenterCrop(224)(image)
        tensor = ts.ToTensor()(image)
        tensor = self.normalize(tensor).reshape(1, 3, 224, 224)
        vector = self.model(tensor).cpu().detach().numpy().flatten()
        return vector
    
image_embedder = ImageEmbedder()


