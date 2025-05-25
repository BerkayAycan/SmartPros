const express = require('express');
const router = express.Router();
const { getPersonalizedSummary } = require('../services/gptService');

router.post('/', async (req, res) => {
  const { userData, drugInfo } = req.body;

  try {
    const summaryText = await getPersonalizedSummary(userData, drugInfo);
    res.json({ summaryText });
  } catch (err) {
    console.error('❌ GPT Hatası:', err);
    res.status(500).json({ summaryText: "Özet alınamadı: " + err.message });
  }
});

module.exports = router;
