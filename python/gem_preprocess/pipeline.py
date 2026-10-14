"""Train coarse centroids from fine centroids and assign documents."""

from pathlib import Path

import numpy as np

from .assignment import initial_membership, tfidf_membership


def write_clusters(path, clusters):
    with path.open("w") as output:
        for documents in clusters:
            output.write(" ".join(map(str, documents)) + "\n")


def assign_to_existing_coarse_centroids(
    codes, lengths, fine_centroids, coarse_centroids, top_r
):
    """Assign documents to an existing coarse codebook.

    Returns the fine-to-coarse labels, initial postings, and the final TF-IDF
    postings.  The inputs are arrays rather than files so library users can
    reuse GEM's assignment logic without staging a second copy of a corpus.
    """
    import faiss

    codes = np.asarray(codes)
    lengths = np.asarray(lengths)
    fine_centroids = np.asarray(fine_centroids, dtype=np.float32)
    coarse_centroids = np.asarray(coarse_centroids, dtype=np.float32)
    if (
        codes.ndim != 1
        or lengths.ndim != 1
        or codes.dtype.kind not in "iu"
        or lengths.dtype.kind not in "iu"
    ):
        raise ValueError("codes and lengths must be one-dimensional integer arrays")
    if np.any(lengths <= 0) or int(lengths.sum()) != len(codes):
        raise ValueError("token/code counts differ")
    if (
        fine_centroids.ndim != 2
        or not np.isfinite(fine_centroids).all()
        or not fine_centroids.shape[1]
    ):
        raise ValueError("invalid fine centroid matrix")
    if (
        coarse_centroids.ndim != 2
        or coarse_centroids.shape[1] != fine_centroids.shape[1]
    ):
        raise ValueError("invalid coarse centroid matrix")
    if codes.size == 0 or codes.min() < 0 or codes.max() >= len(fine_centroids):
        raise ValueError("fine code out of range")
    if top_r <= 0:
        raise ValueError("top_r must be positive")
    quantizer = faiss.IndexFlatIP(fine_centroids.shape[1])
    quantizer.add(coarse_centroids)
    _, labels = quantizer.search(fine_centroids, 1)
    mapped, initial = initial_membership(codes, lengths, labels, len(coarse_centroids))
    return labels, initial, tfidf_membership(mapped, lengths, initial, top_r)


def train_coarse_centroids(
    codes, lengths, fine_centroids, graph_clusters, top_r, iterations, *, gpu=False
):
    """Train GEM's coarse codebook and compute TF-IDF document postings."""
    import faiss

    fine_centroids = np.asarray(fine_centroids, dtype=np.float32)
    if graph_clusters > len(fine_centroids):
        raise ValueError("graph_clusters exceeds fine centroid count")
    trainer = faiss.Kmeans(
        fine_centroids.shape[1],
        graph_clusters,
        niter=iterations,
        gpu=gpu,
        spherical=True,
        verbose=True,
    )
    trainer.train(fine_centroids)
    labels, initial, assigned = assign_to_existing_coarse_centroids(
        codes, lengths, fine_centroids, trainer.centroids, top_r
    )
    return trainer.centroids, labels, initial, assigned


def run(config):
    codes, lengths = [], []
    for shard in range(config.shards):
        c = np.load(config.codes.replace("{shard}", str(shard)), allow_pickle=False)
        n = np.load(config.lengths.replace("{shard}", str(shard)), allow_pickle=False)
        codes.append(c)
        lengths.append(n)
    codes, lengths = np.concatenate(codes), np.concatenate(lengths)
    centers = np.load(config.fine_centroids, allow_pickle=False).astype(np.float32)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    names = (
        "coarse_centroids.npy",
        "coarse_cluster_labels.npy",
        "init_cluster_info.txt",
        "coarse_cluster_info.txt",
    )
    if any((output / name).exists() for name in names):
        raise FileExistsError("encoding output already exists")
    coarse_centroids, labels, initial, assigned = train_coarse_centroids(
        codes,
        lengths,
        centers,
        config.graph_clusters,
        config.top_r,
        config.iterations,
        gpu=config.gpu,
    )
    np.save(output / "coarse_centroids.npy", coarse_centroids)
    np.save(output / "coarse_cluster_labels.npy", labels)
    write_clusters(output / "init_cluster_info.txt", initial)
    write_clusters(output / "coarse_cluster_info.txt", assigned)
