# AI Text Processing Services

This project provides two containerized AI services for advanced text processing: an embedding service using Instructor-XL and a reranking service using BGE Reranker Large. These services work together to enable sophisticated text search and retrieval capabilities.

## 🚀 What This Project Does

### Instructor Service (`instructor/`)
Converts text into high-dimensional vector embeddings using the Instructor-XL model. This service:
- Takes text input and converts it into numerical representations (embeddings)
- Uses instruction-based embeddings for better semantic understanding
- Supports batch processing for efficiency
- Provides L2 normalization for consistent vector magnitudes

### Reranker Service (`reranker/`)
Scores and ranks text passages based on their relevance to a given query using the BGE Reranker Large model. This service:
- Takes a query and multiple candidate passages
- Returns relevance scores for each candidate
- Helps identify the most relevant content for search queries
- Supports batch processing for improved performance

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Instructor    │    │    Reranker     │
│   Service       │    │    Service      │
│                 │    │                 │
│ • Text → Embed  │    │ • Query + Docs  │
│ • Batch Process │    │ • Relevance     │
│ • Normalize     │    │   Scoring       │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
            ┌─────────────────┐
            │   Your App      │
            │                 │
            │ • Search        │
            │ • Retrieval    │
            │ • Ranking       │
            └─────────────────┘
```

## 📋 Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended for optimal performance)
- At least 8GB RAM
- Sufficient disk space for model downloads (~2-3GB per model)

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd cadence-ai-services
```

### 2. Build the Containers

#### Build Instructor Service
```bash
cd instructor
docker build -f Dockerfile.instructor -t instructor-service .
```

#### Build Reranker Service
```bash
cd ../reranker
docker build -f Dockerfile.reranker -t reranker-service .
```

### 3. Run the Services

#### Using Docker Compose (Recommended)
Create a `docker-compose.yml` file in the project root:

```yaml
version: '3.8'
services:
  instructor:
    image: instructor-service
    ports:
      - "8001:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3

  reranker:
    image: reranker-service
    ports:
      - "8002:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Then run:
```bash
docker-compose up -d
```

#### Using Docker Run Commands
```bash
# Run Instructor Service
docker run -d --gpus all -p 8001:8000 --name instructor-service instructor-service

# Run Reranker Service
docker run -d --gpus all -p 8002:8000 --name reranker-service reranker-service
```

## 🔧 API Usage

### Instructor Service (Port 8001)

#### Health Check
```bash
curl http://localhost:8001/healthz
```

#### Generate Embeddings
```bash
curl -X POST "http://localhost:8001/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Represent the document for retrieval: ",
    "texts": ["How to bake a chocolate cake", "Italian pasta recipes"],
    "normalize": true,
    "batch_size": 32
  }'
```

### Reranker Service (Port 8002)

#### Health Check
```bash
curl http://localhost:8002/healthz
```

#### Rerank Documents
```bash
curl -X POST "http://localhost:8002/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to make chocolate cake",
    "candidates": [
      "Chocolate cake recipe with step-by-step instructions",
      "Italian pasta cooking techniques",
      "Chocolate dessert variations and tips"
    ],
    "batch_size": 32
  }'
```

## 📊 Example Workflow

Here's how you might use both services together:

```python
import requests

# 1. Generate embeddings for your documents
documents = [
    "Chocolate cake recipe with step-by-step instructions",
    "Italian pasta cooking techniques", 
    "Chocolate dessert variations and tips"
]

embed_response = requests.post("http://localhost:8001/embed", json={
    "instruction": "Represent the document for retrieval: ",
    "texts": documents,
    "normalize": True
})

embeddings = embed_response.json()["embeddings"]

# 2. Rerank documents based on a query
query = "How to make chocolate cake"

rerank_response = requests.post("http://localhost:8002/rerank", json={
    "query": query,
    "candidates": documents,
    "batch_size": 32
})

scores = rerank_response.json()["scores"]

# 3. Sort documents by relevance
ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
print("Most relevant documents:")
for doc, score in ranked_docs:
    print(f"Score: {score:.3f} - {doc}")
```

## 🔍 Model Information

- **Instructor-XL**: A powerful instruction-tuned embedding model that creates high-quality vector representations
- **BGE Reranker Large**: A cross-encoder model specifically designed for reranking search results

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch sizes or use CPU mode
2. **Model Download Fails**: Check internet connection and Hugging Face access
3. **Service Won't Start**: Verify Docker and GPU drivers are properly installed

### Logs
```bash
# Check service logs
docker logs instructor-service
docker logs reranker-service

# Follow logs in real-time
docker logs -f instructor-service
```

## 📈 Performance Tips

- Use GPU acceleration for better performance
- Adjust batch sizes based on your hardware capabilities
- Consider using multiple instances for high-throughput scenarios
- Monitor memory usage and adjust accordingly

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Support

For issues and questions:
- Check the troubleshooting section above
- Review the API documentation
- Open an issue in the repository
