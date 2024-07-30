# Text Vectorization and Retrieval-Augmented Generation (RAG) Demonstration

This notebook showcases the power of text vectorization and the effectiveness of vector search in processing and understanding natural language data. Additionally, it demonstrates the core steps of Retrieval-Augmented Generation (RAG) using various tools and models.

## Dependencies

- **BAAI/bge-large-zh-v1.5**: Embedding model with a 1024-dimensional embedding space.
- **google/flan-t5-base**: Text generation model.
- **sklearn.metrics.pairwise.cosine_similarity**: Function to compute cosine similarity between vectors.
- **sklearn.decomposition.PCA**: Tool to perform Principal Component Analysis (PCA) for dimensionality reduction.
- **matplotlib**: Library for creating 2D and 3D plots.
- **mplcursors**: Interactive data cursor for `matplotlib`.

## Content Overview

### I. Define Functions
Three key functions are defined to streamline the analysis:
- `plot_2D`: Generates 2D scatter plots of vectorized text data.
- `get_embedding`: Converts text into numerical vectors using the embedding model.
- `compare`: Compares different text examples based on their vector representations.

### II. Define Examples
Introduces 12 examples across 4 distinct categories: Technology, Finance, World Facts, and Stock Market.

### III. Embedding
Vectorizes the examples using the embedding model, transforming text data into high-dimensional vectors.

### IV. Apply PCA
Reduces the dimensionality of embeddings from 1024 to 3 dimensions using PCA for visualization.

### V. Plot 2D
Creates 2D plots to observe the separation between different categories achieved through vectorization.

### VI. Plot 3D
Extends visualization to 3D plots, providing another perspective on category separation.

### VII. Computing Cosine Similarity
Computes cosine similarity between selected examples to illustrate the effectiveness of vector search in identifying similar text data.

### VIII. Simple Retrieval-Augmented Generation (RAG)
1. **Vectorize User Query**: Vectorizes the user's query.
2. **Vector Search**: Searches the vector database for the most similar embeddings.
3. **Retrieve Top Results**: Retrieves the top `n` results based on cosine similarity.
4. **LLM Input Preparation**: Prepares the input for the Language Model (LLM) using the retrieved content and the user's query.
5. **Generate Response**: Uses the `google/flan-t5-base` model to generate a response from the retrieved content.

## Installation
```bash
pip install -q -U FlagEmbedding mplcursors ipympl
