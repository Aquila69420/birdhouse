const express = require('express');
const bodyParser = require('body-parser');
const { exec } = require('child_process');

const app = express();
app.use(bodyParser.json());

app.post('/generate-api-key', (req, res) => {
    const { walletAddress } = req.body;

    // Ensure walletAddress is provided
    if (!walletAddress) {
        return res.status(400).json({ success: false, message: 'Wallet address is required.' });
    }

    // Construct the curl command
    const command = `curl -X 'POST' 'https://rag-chat-ml-backend-dev.flock.io/api_key/generate' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{"wallet_address": "${walletAddress}"}'`;

    // Execute the command
    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error: ${stderr}`);
            return res.status(500).json({ success: false, message: 'Failed to generate API key.', error: stderr });
        }

        // Parse JSON response from stdout
        let apiResponse;
        try {
            apiResponse = JSON.parse(stdout);
        } catch (err) {
            console.error('Failed to parse response:', err);
            return res.status(500).json({ success: false, message: 'Error parsing API response.' });
        }

        // Return success response with API key
        return res.json({ success: true, data: apiResponse });
    });
});

app.listen(3000, () => {
    console.log('Server running on http://localhost:3000');
});
