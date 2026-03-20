require('dotenv').config();

const express = require('express');
const cors = require('cors');
const pipelineRoutes = require('./routes/pipeline');

const app = express();
const PORT = process.env.PORT || 5000;

const allowedOrigins = new Set([
  'http://localhost:5173',
  'http://localhost:8080',
  'http://127.0.0.1:5173',
  'http://127.0.0.1:8080',
]);

app.use(
  cors({
    // Echo back the request Origin for allowed dev URLs.
    origin: (origin, callback) => {
      if (!origin) return callback(null, true);
      if (allowedOrigins.has(origin)) return callback(null, origin);
      return callback(null, false);
    },
    credentials: true,
  })
);
app.use(express.json());

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', service: 'NAVIER Backend' });
});

app.use('/api', pipelineRoutes);

app.listen(PORT, () => {
  console.log(`NAVIER Backend listening on http://localhost:${PORT}`);
});
