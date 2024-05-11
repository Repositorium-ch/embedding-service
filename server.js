import express from 'express';
import { AutoTokenizer } from '@xenova/transformers';
import { env, pipeline } from '@xenova/transformers';
import cors from 'cors';
import crypto from 'crypto';

// Load environment variables
if(!process.env.MODEL) {
    process.env.MODEL = 'Xenova/all-MiniLM-L6-v2'
}

// Load API Key
const API_KEY = process.env.API_KEY || crypto.randomBytes(32).toString('hex');

if(!process.env.API_KEY) {
    console.log('API Key not found. Generated a new one for you: ', API_KEY);
}


// Middleware for Bearer token validation
const authenticateBearerToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
  
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ message: 'Unauthorized: Missing Bearer token' });
    }
  
    const token = authHeader.split(' ')[1];
    if (token !== API_KEY) {
      return res.status(403).json({ message: 'Forbidden: Invalid Bearer token' });
    }
  
    next();
  };

// Specify a custom location for models (defaults to '/models/').
env.localModelPath = '/models/';
// Disable the loading of remote models from the Hugging Face Hub:
//env.allowRemoteModels = false;

// Set location of .wasm files. Defaults to use a CDN.
env.backends.onnx.wasm.wasmPaths = '/models/';

const app = express();

// CORS Settings
const corsOptions = {
    origin: 'http://localhost:3000',
    optionsSuccessStatus: 200
}
app.use(authenticateBearerToken);
app.use(cors(corsOptions));
app.use(express.json()); // Parse JSON request bodies

// Load model asynchronously
let extractor;
pipeline('feature-extraction', process.env.MODEL)
    .then(model => {
        extractor = model;
        console.log('Model loaded successfully');
    })
    .catch(err => {
        console.error('Failed to load model:', err);
        process.exit(1);
    });

const tokenizer = await AutoTokenizer.from_pretrained(process.env.MODEL);


// Create a POST endpoint to receive the audio file
app.post('/v1/embeddings', async (req, res) => {
    try {
        console.log("### Initiating Embedding Request.");

        // Validate input
        if (!req.body.input) {
            return res.status(400).json({ error: 'Invalid input: input' });
        }

        let strings;
        if (Array.isArray(req.body.input)) {
            strings = req.body.input;
        } else if (typeof req.body.input === 'string') {
            strings = [req.body.input];
        } else {
            return res.status(400).json({ error: 'Invalid input: input should be a string or an array of strings' });
        }

        console.log("Strings: " + strings);

        // Compute features
        let output = await extractor(strings, { pooling: 'mean', normalize: true });

        let response = {
            "object": "list",
            "model": process.env.MODEL,
            "data": output.tolist(),
        }

        // Convert Tensor to JS list
        //output = output.tolist();

        // Send the features back
        res.status(200).json(response);
    } catch (err) {
        console.error('Error processing request:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Start the server
app.listen(3000, () => console.log('Server started on port 3000'));