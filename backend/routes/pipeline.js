const express = require('express');
const axios = require('axios');

const router = express.Router();
const AI_URL = process.env.NAVIER_AI_URL || 'http://localhost:8001';

async function forwardJson(res, promise) {
  try {
    const r = await promise;
    res.status(r.status).json(r.data);
  } catch (err) {
    const status = err.response?.status ?? 502;
    const data = err.response?.data ?? { error: err.message || 'Upstream error' };
    res.status(status).json(typeof data === 'object' ? data : { error: String(data) });
  }
}

router.post('/analyze', (req, res) => {
  forwardJson(
    res,
    axios.post(`${AI_URL}/analyze`, req.body, {
      headers: { 'Content-Type': 'application/json' },
      validateStatus: () => true,
    })
  );
});

router.post('/simulate_leak', (req, res) => {
  forwardJson(
    res,
    axios.post(`${AI_URL}/simulate_leak`, req.body, {
      headers: { 'Content-Type': 'application/json' },
      validateStatus: () => true,
    })
  );
});

router.get('/sensor_status', (req, res) => {
  forwardJson(
    res,
    axios.get(`${AI_URL}/sensor_status`, {
      params: req.query,
      validateStatus: () => true,
    })
  );
});

router.get('/model/info', (req, res) => {
  forwardJson(
    res,
    axios.get(`${AI_URL}/model/info`, {
      validateStatus: () => true,
    })
  );
});

module.exports = router;
