import express from 'express';
import { env, pipeline } from '@xenova/transformers';
import cors from 'cors';

// Load environment variables
if(!process.env.MODEL) {
    process.env.MODEL = 'Xenova/all-MiniLM-L6-v2'
}

const app = express();

// CORS Settings
const corsOptions = {
    origin: 'http://localhost:3000',
    optionsSuccessStatus: 200
}

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

// Create a POST endpoint to receive the audio file
app.post('/api/embed', async (req, res) => {
    try {
        console.log("### Initiating Embedding Request.");

        // Validate input
        if (!req.body.strings || !Array.isArray(req.body.strings)) {
            return res.status(400).json({ error: 'Invalid input: strings' });
        }

        let strings = req.body.strings;
        console.log("Strings: " + strings);

        // Compute features
        let output = await extractor(strings, { pooling: 'mean', normalize: true });

        let response = {
            "output": output,
            "status": "success",
            "model": process.env.MODEL,
            "embeddings": output.tolist()
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

// Serve static files from /public (at the moment only index.html)
app.use(express.static('public'));

// Start the server
app.listen(3000, () => console.log('Server started on port 3000'));