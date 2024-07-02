import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import PCA
import json
import glob
import numpy as np
from scipy.spatial.distance import pdist, squareform

sentences = []
sentences_full = []

for f in glob.glob("./result/prompt_*.json"):
    with open(f, "r") as r:
        for x in json.loads(r.read()):
            sentences.append(x["user"])
            sentences_full.append(x)

tagged_data = [
    TaggedDocument(words=sentence.split(), tags=[str(i)])
    for i, sentence in enumerate(sentences)
]

model = Doc2Vec(vector_size=100, min_count=1, epochs=50)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)


def plot(sentences: list[str], save: str):
    sentence_vectors = [model.dv[str(i)] for i in range(len(sentences))]

    pca = PCA(n_components=2)
    result = pca.fit_transform(sentence_vectors)

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(
        result[:, 0], result[:, 1], c=range(len(sentences)), cmap="viridis"
    )

    plt.colorbar(
        scatter,
    )
    plt.tight_layout()
    plt.savefig(save)


sentence_vectors = [model.dv[str(i)] for i in range(len(sentences))]
sentence_vectors = np.array(sentence_vectors)
distances = pdist(sentence_vectors)
distance_matrix = squareform(distances)

percentile = 0.05
threshold = np.percentile(distances, percentile)


def get_distant_pairs(distance_matrix, threshold):
    pairs = set()
    n = distance_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] < threshold:
                pairs.add(i)
                pairs.add(j)
    return pairs


distant_indexes = get_distant_pairs(distance_matrix, threshold)

sentences_new = []
sentences_new_q = []
for i, s in enumerate(sentences_full):
    if not i in distant_indexes:
        sentences_new.append(s)
        sentences_new_q.append(s["user"])
    else:
        print(s["user"])

with open("./result/filtered.json", "w") as w:
    w.write(json.dumps(sentences_new, ensure_ascii=False))

plot(sentences, "plot_before.png")
plot(sentences_new_q, "plot_after.png")
