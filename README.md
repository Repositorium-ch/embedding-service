# Minimal Embedding Service
This is a minimal embedding service that allows you to embed any content. It is built using Express.js transformers.js

## Installation
1. Clone the repository
2. Run `docker build -t embedding_server .`
3. Run `docker run -v $(pwd)/volume:/usr/src/app/node_modules/@xenova/transformers/.cache -p 3000:3000 -e MODEL='YourModel' embedding_server`

Example: `docker run -v $(pwd)/volume:/usr/src/app/node_modules/@xenova/transformers/.cache -p 3000:3000 -e MODEL='Snowflake/snowflake-arctic-embed-xs' embedding_server`

If you do not set a Model in the docker run commmand (for example `docker run -p 3000:3000 embedding_server`) the model `Xenova/all-MiniLM-L6-v2` will be used automatically.
## Usage
To embed a content, make a POST request to `http://localhost:3000/api/embed` with the following body:
```json
{
     "strings": ["sentence 1", "sentence 2", "sentence 3"] 
}
```

You can also specify the model to use by setting the `MODEL` environment variable. The default model is `distilbert-base-uncased`.
It has to be a ONNX model, compatible with transformers.js.

**There is also a simple client-example served at the root of the server. You can access it by going to `http://localhost:3000/`**
