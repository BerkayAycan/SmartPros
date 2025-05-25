const express = require('express');
const router = express.Router();
const SummaryHistory = require('../models/SummaryHistory');

// Özet geçmişi listeleme
router.get('/:userId', async (req, res) => {
  try {
    const history = await SummaryHistory.find({ userId: req.params.userId }).sort({ createdAt: -1 });
    res.json(history);
  } catch (err) {
    res.status(500).json({ error: 'Sunucu hatası' });
  }
});

// Yeni özet kaydı ekleme
router.post('/', async (req, res) => {
  const { userId, drugName, summaryText } = req.body;
  try {
    const record = new SummaryHistory({ userId, drugName, summaryText });
    await record.save();
    res.json({ message: 'Özet geçmişe kaydedildi' });
  } catch (err) {
    res.status(500).json({ error: 'Kayıt başarısız' });
  }
});

module.exports = router;
