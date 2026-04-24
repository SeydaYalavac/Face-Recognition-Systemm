import os
import numpy as np
import matplotlib.pyplot as plt


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def dynamic_threshold_from_brightness(brightness_value, min_threshold=0.45, max_threshold=0.68):
    """Adjust cosine similarity threshold using ambient brightness."""
    normalized = np.clip(brightness_value / 255.0, 0.0, 1.0)
    threshold = max_threshold - (max_threshold - min_threshold) * normalized
    return float(np.clip(threshold, min_threshold, max_threshold))


def find_best_match(query_embedding, database_embeddings, database_labels, ambient_brightness=None):
    """Find the best match using cosine similarity and dynamic thresholding."""
    if ambient_brightness is None:
        threshold = 0.55
    else:
        threshold = dynamic_threshold_from_brightness(ambient_brightness)

    best_score = -1.0
    best_label = None

    for emb, label in zip(database_embeddings, database_labels):
        score = cosine_similarity(query_embedding, emb)
        if score > best_score:
            best_score = score
            best_label = label

    if best_score >= threshold:
        return best_label, best_score, threshold
    return None, best_score, threshold


def plot_similarity_heatmap(embeddings, labels, title='Cosine Similarity Heatmap', output_path=None):
    """Visualize the similarity matrix for embeddings."""
    matrix = np.zeros((len(embeddings), len(embeddings)), dtype=np.float32)
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(matrix, cmap='viridis', vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(cax, ax=ax, label='Cosine Similarity')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()

    return matrix


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize cosine similarity between face embeddings.')
    parser.add_argument('--labels', nargs='+', required=True, help='Labels for each embedding')
    parser.add_argument('--output', default=None, help='Optional path to save heatmap image')
    args = parser.parse_args()

    sample_vectors = [np.random.randn(128).astype(np.float32) for _ in args.labels]
    plot_similarity_heatmap(sample_vectors, args.labels, output_path=args.output)
