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
    os.mkdir(DATA_DIR)

    if not os.path.exists(IMAGENET_DIR):
        if not os.path.exists(IMAGENET_ZIP):
            r = requests.get(IMAGENET_URL)
            with open(IMAGENET_ZIP, 'wb') as f:
                f.write(r.content)
        with zipfile.ZipFile(IMAGENET_ZIP, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

# download_data()

from torchvision import datasets
import random

image_classes = set(random.sample(range(200), 5) )

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

from tqdm import tqdm
import pandas as pd

df = pd.DataFrame()
df["id"] = image_file_names
# df["id"] = [
#     file_name.split(IMAGENET_DIR)[-1] for file_name in image_file_names
# ]
df["values"] = [
    image_embedder.embed(file_name) for file_name in tqdm(image_file_names)
]
df = df.sample(frac = 1)

print(df.head(2))

cutoff = int(len(df)*0.9)
item_df, query_df = df[:cutoff], df[cutoff:]


#Create a pinecone vector index

index_name = 'arian-index'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, metric = 'euclidean', shards=1, dimension = 768)

index = pinecone.Index(index_name)

data_list = []

# Loop through each row of the DataFrame
for idx, row in item_df.iterrows():
    # Extract values from each row
    embedding_id = row['id']
    embedding = row['values']
    # image_file_name = row['metadata']
    
    # Create a dictionary for the current row
    data_dict = {
        'id': embedding_id,
        'values': embedding,
        # 'image_file_name': image_file_name
    }
    
    # Append the dictionary to the data_list
    data_list.append(data_dict)
    
acks = index.upsert(vectors = data_list)

import time

start = time.perf_counter()
res = index.query(query_df["values"], batch_size =100)
end = time.perf_counter()
print(f'Throughput is {int(len(query_df)/end-start)} queries/sec')

for i in range(80,90):
    print(f'Query {i+1} and search results')
    show_image(query_df["id"].iloc[i])
    show_images_horizentally(
        [IMAGENET_DIR + embedding_id for embedding_id in res[i].ids]
    )
    print("-" * 100)