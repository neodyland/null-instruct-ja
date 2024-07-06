import os
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from tqdm import tqdm
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load and prepare data
sentences = []
sentences_full = []

RE_TOO_MANY_ALPHABETS = re.compile(r"[a-zA-Z]{5,}")


def filter_ok(text: str):
    return RE_TOO_MANY_ALPHABETS.match(text) == None


for f in glob.glob("./result/prompt_*.json"):
    with open(f, "r") as r:
        for x in json.loads(r.read()):
            if "reject" in x:
                if x["reject"] == x["model"]:
                    continue
            if filter_ok(x["user"]) and filter_ok(x["model"]):
                sentences.append(x["user"])
                sentences_full.append(x)

with open("./result/filtered.json", "w") as w:
    json.dump(sentences_full, w, ensure_ascii=False)

print("Data loaded")

# Initialize the model
model = SentenceTransformer("intfloat/multilingual-e5-large")
model.cuda()
model.eval()

print("Model loaded")

# Encode sentences in batches
batch_size = 32
vectors = []

for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
    batch = sentences[i : i + batch_size]
    batch_vectors = model.encode(batch, show_progress_bar=False)
    vectors.append(batch_vectors)

vectors = np.vstack(vectors)
print("Encoding completed")

# Perform t-SNE
tsne = TSNE(n_components=2, metric="cosine", n_jobs=-1, random_state=42)
tsne_vectors = tsne.fit_transform(vectors)
print("t-SNE completed")

# Plot results
plt.figure(figsize=(12, 10))
plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], s=5, alpha=0.5)
plt.title("テキストの分布")
plt.tight_layout()
plt.savefig("plot.png", dpi=300)
print("Plot saved as plot.png")
