const express = require('express');
const bodyParser = require('body-parser');
const { exec } = require('child_process');
const cors = require("cors")

const app = express();
app.use(bodyParser.json());
app.use(cors({origin: "*"}))


app.post('/generate-api-key/', (req, res) => {
    console.log("Is Working");
    const { walletAddress } = req.body;

    if (!walletAddress) {
        return res.status(400).json({ success: false, message: 'Wallet address is required.' });
    }

    const command = `curl -X 'POST' 'https://rag-chat-ml-backend-dev.flock.io/api_key/generate' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{"wallet_address": "${walletAddress}"}'`;

    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error: ${stderr}`);
            return res.status(500).json({ success: false, message: 'Failed to generate API key.', error: stderr });
        }

        let apiResponse;
        try {
            apiResponse = JSON.parse(stdout);
        } catch (err) {
            console.error('Failed to parse response:', err);
            return res.status(500).json({ success: false, message: 'Error parsing API response.' });
        }

        return res.json({ success: true, data: apiResponse });
    });
});

app.listen(12345, () => {
    console.log('Server running on http://localhost:12345');
});
