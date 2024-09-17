import express from 'express';
import { env, pipeline, AutoTokenizer, AutoConfig } from '@xenova/transformers';
import cors from 'cors';
import crypto from 'crypto';

// Load environment variables
const MODEL = process.env.MODEL || 'Xenova/all-MiniLM-L6-v2';
const PORT = process.env.PORT || 3000;
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || 'http://localhost:3000';

// Load API Key
const API_KEY = process.env.API_KEY || crypto.randomBytes(32).toString('hex');

if (!process.env.API_KEY) {
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

// Disable the loading of remote models from the Hugging Face Hub:
env.allowRemoteModels = false;

const app = express();

// CORS Settings
app.use(cors({ origin: ALLOWED_ORIGIN, optionsSuccessStatus: 200 }));
app.use(authenticateBearerToken);
app.use(express.json()); // Parse JSON request bodies

// Load model, tokenizer, and config asynchronously
let extractor;
let tokenizer;
let MAX_TOKENS;

const loadModelAndConfig = async () => {
    try {
        const [model, tokenizerInstance, config] = await Promise.all([
            pipeline('feature-extraction', MODEL),
            AutoTokenizer.from_pretrained(MODEL),
            AutoConfig.from_pretrained(MODEL)
        ]);

        extractor = model;
        tokenizer = tokenizerInstance;

        // Set MAX_TOKENS based on config
        MAX_TOKENS = config.max_position_embeddings 
            ? config.max_position_embeddings - 2  // Subtract 2 for start and end tokens
            : 512;  // Default to 512 if not specified in config

        console.log(`Model and tokenizer loaded successfully. MAX_TOKENS set to ${MAX_TOKENS}`);
    } catch (err) {
        console.error('Failed to load model, tokenizer, or config:', err);
        process.exit(1);
    }
};

// Function to truncate text to MAX_TOKENS with detailed information
const truncateToMaxTokens = async (text) => {
    console.log(`Truncating text: "${text.substring(0, 50)}..."`);
    
    let originalTokenCount, truncatedTokenCount, tokensRemoved;

    try {
        const encoded = await tokenizer(text, { truncation: false });
        console.log('Encoded:', encoded); // Debugging line

        // Get the token count from the dimensions of the input_ids tensor
        originalTokenCount = encoded.input_ids.dims[1]; // Since dims is [batch_size, seq_length]

        console.log(`Original token count: ${originalTokenCount}`);
        console.log(`MAX_TOKENS: ${MAX_TOKENS}`);

        if (originalTokenCount <= MAX_TOKENS) {
            console.log('No truncation needed');
            return {
                text: text,
                truncated: false,
                originalTokenCount: originalTokenCount,
                truncatedTokenCount: originalTokenCount,
                tokensRemoved: 0,
                prompt_tokens: originalTokenCount
            };
        }

        console.log('Truncation needed');
        const truncatedEncoded = await tokenizer(text, { truncation: true, maxLength: MAX_TOKENS });
        console.log('Truncated Encoded:', truncatedEncoded); // Debugging line

        truncatedTokenCount = truncatedEncoded.input_ids.dims[1]; // Get the truncated token count

        // Convert BigInt64Array to array of numbers
        const tokenIds = Array.from(truncatedEncoded.input_ids.data, BigInt).map(Number);

        // Decode the tokens back to text
        const truncatedText = await tokenizer.decode(tokenIds);

        tokensRemoved = originalTokenCount - truncatedTokenCount;

        console.log(`Truncated token count: ${truncatedTokenCount}`);
        console.log(`Tokens removed: ${tokensRemoved}`);

        return {
            text: truncatedText,
            truncated: true,
            originalTokenCount: originalTokenCount,
            truncatedTokenCount: truncatedTokenCount,
            tokensRemoved: tokensRemoved,
            prompt_tokens: truncatedTokenCount
        };
    } catch (error) {
        console.error('Error in tokenization:', error);
        // Fallback to string length as a rough estimate
        const stringLength = text.length;
        return {
            text: text,
            truncated: false,
            originalTokenCount: stringLength,
            truncatedTokenCount: stringLength,
            tokensRemoved: 0,
            error: 'Tokenization failed, using string length as estimate',
            prompt_tokens: stringLength
        };
    }
};

// Create a POST endpoint to receive the text and return embeddings
// Create a POST endpoint to receive the text and return embeddings
app.post('/v1/embeddings', async (req, res) => {
    try {
        console.log("### Initiating Embedding Request.");

        // Validate input
        if (!req.body.input) {
            return res.status(400).json({ error: 'Invalid input: input is required' });
        }

        let strings;
        if (Array.isArray(req.body.input)) {
            strings = req.body.input;
        } else if (typeof req.body.input === 'string') {
            strings = [req.body.input];
        } else {
            return res.status(400).json({ error: 'Invalid input: input should be a string or an array of strings' });
        }

        console.log("### Received Strings, now truncating and computing embeddings...");
        console.time("Computation Time");

        // Truncate strings if necessary
        const truncationResults = await Promise.all(strings.map(truncateToMaxTokens));
        const truncatedStrings = truncationResults.map(result => result.text);

        // Compute total tokens
        const totalPromptTokens = truncationResults.reduce((sum, result) => sum + result.prompt_tokens, 0);

        // Compute features
        let output = await extractor(truncatedStrings, { pooling: 'mean', normalize: true });

        output = output.tolist();

        let data = output.map((embedding, index) => ({
            "object": "embedding",
            "index": index,
            "embedding": embedding,
        }));

        let response = {
            "object": "list",
            "data": data,
            "model": MODEL,
            "usage": {
                "prompt_tokens": totalPromptTokens,
                "total_tokens": totalPromptTokens // Assuming no additional tokens are used
            },
            //Optionally include truncation results for debugging:
            "truncation_results": truncationResults.map(result => ({
                truncated: result.truncated,
                originalTokenCount: result.originalTokenCount,
                truncatedTokenCount: result.truncatedTokenCount,
                tokensRemoved: result.tokensRemoved,
                error: result.error
            }))
        };

        console.timeEnd("Computation Time");
        console.log("### Embedding Request Completed.");
        // Optionally log truncation results
        // console.log("Truncation Results:", JSON.stringify(response.truncation_results, null, 2));

        // Send the features back
        res.status(200).json(response);
    } catch (err) {
        console.error('Error processing request:', err);
        res.status(500).json({ error: 'Internal server error', message: err.message });
    }
});

// Initialize the model and config before starting the server
loadModelAndConfig().then(() => {
    app.listen(PORT, () => {
        console.log(`Server is running on port ${PORT}`);
    });
}).catch(err => {
    console.error('Failed to initialize the server:', err);
    process.exit(1);
});