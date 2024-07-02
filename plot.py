import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import PCA
import json
import glob

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


with open("./result/filtered.json", "w") as w:
    w.write(json.dumps(sentences_full, ensure_ascii=False))

plot(sentences, "plot.png")
