const mongoose = require('mongoose');

const SummaryHistorySchema = new mongoose.Schema({
  userId: { type: String, required: true },
  drugName: { type: String, required: true },
  summaryText: { type: String, required: true },
  createdAt: { type: Date, default: Date.now },
});

module.exports = mongoose.model('SummaryHistory', SummaryHistorySchema);
