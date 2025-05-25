const mongoose = require('mongoose');
const fs = require('fs');
const csv = require('csv-parser');
require('dotenv').config();

const DrugSchema = new mongoose.Schema({
  name: String,
  pdfUrl: String,
});

const Drug = mongoose.model('Drug', DrugSchema);

async function loadCsvData() {
  await mongoose.connect(process.env.MONGODB_URI);
  console.log('✅ MongoDB Bağlandı');

  const drugs = [];

  fs.createReadStream('ai/DrugsData_cleaned.csv') // CSV yolu doğru olmalı
    .pipe(csv({ separator: ',' }))
    .on('data', (row) => {
      if (row['ilaç adı'] && row['küb PDF']) {
        drugs.push({
          name: row['ilaç adı'],
          pdfUrl: row['küb PDF'],
        });
      }
    })
    .on('end', async () => {
      await Drug.insertMany(drugs);
      console.log('✅ Veriler MongoDB\'ye başarıyla yüklendi!');
      mongoose.disconnect();
    });
}

loadCsvData().catch(err => {
  console.error(err);
  mongoose.disconnect();
});
