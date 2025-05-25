const express = require('express');
const router = express.Router();
const Drug = require('../models/Drug');

router.get('/', async (req, res) => {
    const query = req.query.query || '';
    if (!query) return res.json([]);

    try {
        const results = await Drug.find({
            name: { $regex: query, $options: 'i' }
        }).limit(10); // ilk 10 eþleþme

        const names = results.map(d => d.name);
        res.json(names);
    } catch (err) {
        console.error("Autocomplete Hatasý:", err);
        res.status(500).json([]);
    }
});

module.exports = router;
