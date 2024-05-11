# Minimal Embedding Service
This is a minimal embedding service that allows you to embed any content. It is built using Express.js and transformers.js.

## Installation
1. Clone the repository
2. Run `docker build -t embedding_server .`
3. Run `docker run -v $(pwd)/volume:/usr/src/app/node_modules/@xenova/transformers/.cache -p 3000:3000 -e MODEL='YourModel' -e API_KEY='YourSuperSecureApiKey' embedding_server`

Example: `docker run -v $(pwd)/volume:/usr/src/app/node_modules/@xenova/transformers/.cache -p 3000:3000 -e MODEL='Snowflake/snowflake-arctic-embed-xs' embedding_server`

If you do not set an API_KEY in the docker run commmand (for example `docker run -p 3000:3000 embedding_server`) a random API_KEY will be generated automatically and printed to the console.

If you do not set a Model in the docker run commmand (for example `docker run -p 3000:3000 embedding_server`) the model `Xenova/all-MiniLM-L6-v2` will be used automatically.
## Usage
To embed content, make a POST request to `http://localhost:3000/v1/embedddings` with the following body:
```json
{
  "input": "Your text string goes here"
}
```

or 

```json
{
  "input": ["Your", "text", "array", "goes", "here"]
}
```

curl example:

```curl
curl http://localhost:3000/v1/embeddings \ 
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YourSuperSecureApiKey" \
  -d '{
    "input": "Your text string goes here",
  }'
```

You can also specify the model to use by setting the `MODEL` environment variable. The default model is `Xenova/all-MiniLM-L6-v2`.
The model has to be a ONNX model, compatible with transformers.js.
