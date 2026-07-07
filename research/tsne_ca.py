import matplotlib.pyplot as plt
import numpy as np
import pathlib
import argparse
import random
from sklearn.manifold import TSNE


def color_generator(art_name):
    random.seed(art_name)
    r = random.random()
    g = random.random()
    b = random.random()
    return (r, g, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=pathlib.Path, help="Path to the directory containing the embeddings")
    
    args = parser.parse_args()
    
    npy_files = list(args.src_dir.rglob("*.npy"))
    
    list_names = [f.parents[0].name for f in npy_files]
    
    embeddings = np.stack([np.load(f) for f in npy_files])
    
    tsne = TSNE(n_components=2, random_state=0, perplexity=30, verbose=1)
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    colors=[color_generator(name) for name in list_names]
    
    plt.title("t-SNE on Modality Projection Embeddings from Prato della Valle Dataset")
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=colors)
    plt.show()

if __name__ == "__main__":
    main()