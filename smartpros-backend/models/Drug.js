const mongoose = require('mongoose');

const DrugSchema = new mongoose.Schema({
  name: String,
  pdfUrl: String
});

module.exports = mongoose.model('Drug', DrugSchema);
