# Health-Bot using LLama 2: Medical Chatbot 

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Code Examples](#code-examples)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains the source code for Health-Bot, a specialized chatbot aimed at providing reliable and instant medical information. Utilizing the power of advanced NLP models and search algorithms, the chatbot can dig through a large database of medical documents to provide accurate and quick responses.

## Features

- **Pre-trained and Quantized LLama 2 Model**: Ensures high-quality medical responses.
- **CTransformers Python Bindings**: Fast and efficient low-level access to the LLama model.
- **Sentence Transformer Embeddings (All MiniLM v6)**: Robust document embeddings for accurate search and retrieval.
- **Searchable Vector Store Options**:
    - **Chroma DB**: For high-capacity, large-scale applications.
    - **Faiss CPU**: For quick and memory-efficient applications.
    - **Qdrant**: An alternative vector storage option.

## Architecture


## Installation

```bash
# Clone the repository
git clone https://github.com/your_username/health-bot.git

# Navigate to the project directory
cd health-bot

# Install required dependencies
pip install -r requirements.txt
```

## File Structure

```
.
├── digest.py                  # Converts PDFs to vector embeddings
├── model.py                   # Main logic for query handling and response
├── data/                      # Store your medical PDFs here
├── vectorstores/              
│   ├── chroma_db/             # Chroma DB storage
│   ├── faiss_db/              # Faiss CPU storage
│   └── qdrant/                # Qdrant storage
├── requirements.txt           # Dependency list
└── README.md
```

## Usage

### Creating the Vector Database

The first step is to populate the vector database with medical documents for the chatbot to search.

```bash
# Run this script to populate the vector database
python digest.py
```

### Start the Chatbot

The chatbot can be initiated using the following command.

```bash
# Start the chatbot
python model.py
```

## Code Examples

### Creating the Vector Database

In `digest.py`, the `create_vector_db()` function creates a searchable vector database from PDFs stored in the `data/` directory.

```python
# digest.py
if __name__ == '__main__':
    create_vector_db()
```

### Setting up the QA Bot

In `model.py`, the `query_response` function handles the main logic for generating a response based on a query.

```python
# model.py
def query_response(query_text):
    query_chain = setup_qa_bot()
    processed_response = query_chain({'query': query_text})
    return processed_response
```

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

---

Feel free to modify the README.md as per the needs of your project.