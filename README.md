# Embedding Vector Space Viewer

Interactive 3D visualization of word embedding clusters using Three.js and t-SNE.

![preview](https://img.shields.io/badge/words-738-blue) ![preview](https://img.shields.io/badge/clusters-18-green) ![preview](https://img.shields.io/badge/method-t--SNE-orange)

## Quick Start

```bash
# 1. Generate embeddings (only needed once, or to regenerate)
pip install numpy scikit-learn
python backend/generate_embeddings.py

# 2. Launch the viewer
python serve.py --port 8080

# 3. Open in browser
# http://localhost:8080
```

## Controls

| Action          | Input               |
|-----------------|---------------------|
| Orbit           | Click + drag        |
| Zoom            | Scroll wheel        |
| Inspect word    | Hover               |
| Filter cluster  | Click legend item   |
| Search          | Type in search box  |

## Using Real GloVe Embeddings

The default setup uses synthetic clustered embeddings. To swap in real GloVe vectors:

1. Download GloVe from [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/) (the `glove.6B.zip` file, 50d variant is sufficient)
2. Replace the embedding generation in `backend/generate_embeddings.py`:

```python
import numpy as np
from sklearn.manifold import TSNE
import json

# Load GloVe
vectors = {}
with open("data/glove.6B.50d.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vec = np.array([float(x) for x in parts[1:]])
        vectors[word] = vec

# Take top N most common words
words = list(vectors.keys())[:1000]
matrix = np.array([vectors[w] for w in words])

# t-SNE to 3D
tsne = TSNE(n_components=3, perplexity=30, max_iter=1000, random_state=42)
coords = tsne.fit_transform(matrix)

# Normalize to [-50, 50]
for d in range(3):
    mn, mx = coords[:, d].min(), coords[:, d].max()
    coords[:, d] = ((coords[:, d] - mn) / (mx - mn) - 0.5) * 100

# Export (categories would be auto-detected or manually assigned)
data = {"words": [], "categories": ["default"], "color_map": {"default": "#4a7cff"}}
for i, word in enumerate(words):
    data["words"].append({
        "word": word,
        "x": float(coords[i, 0]),
        "y": float(coords[i, 1]),
        "z": float(coords[i, 2]),
        "category": "default",
        "color": "#4a7cff"
    })

with open("data/embeddings_3d.json", "w") as f:
    json.dump(data, f)
```

3. Re-run `python backend/generate_embeddings.py` and refresh the browser.

## Project Structure

```
embedding-viz/
├── backend/
│   └── generate_embeddings.py   # Embedding generation + t-SNE
├── frontend/
│   └── index.html               # Three.js visualization (self-contained)
├── data/
│   └── embeddings_3d.json       # Pre-computed 3D coordinates
├── serve.py                     # Local dev server
└── README.md
```

## Dependencies

- **Python**: numpy, scikit-learn (for generation only)
- **Frontend**: Three.js r128 (loaded from CDN, no build step)
- **Optional**: gensim (for loading pre-trained GloVe/Word2Vec models)
