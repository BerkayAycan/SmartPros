const fs = require('fs');
const csv = require('csv-parser');
const mongoose = require('mongoose');
const dotenv = require('dotenv');
const Drug = require('./models/Drug');

dotenv.config();

mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log("✅ MongoDB bağlantısı başarılı"))
  .catch(err => console.error("❌ MongoDB bağlantı hatası:", err));

const results = [];

fs.createReadStream("C:\\Users\\fatih\\Downloads\\DrugsData_cleaned.csv")
  .pipe(csv())
  .on('data', (data) => {
    // Tüm başlıkları küçük harf yap ve boşlukları kaldır
    const normalizedData = {};
    for (let key in data) {
      const newKey = key.trim().toLowerCase();
      normalizedData[newKey] = data[key];
    }

    const name = (normalizedData["ilaç adı"] || "").replace(/['"“”]/g, "").trim();
    const pdfUrl = (normalizedData["küb pdf"] || "").trim();

    if (name && pdfUrl) {
      results.push({ name, pdfUrl });
    }
  })
  .on('end', async () => {
    try {
      if (results.length === 0) {
        console.log("❗ CSV'den uygun veri bulunamadı. Başlık eşleşmesini çözemedik.");
      } else {
        await Drug.insertMany(results);
        console.log(`✅ ${results.length} ilaç MongoDB'ye yüklendi.`);
      }
      mongoose.disconnect();
    } catch (err) {
      console.error("❌ Veritabanına kayıt sırasında hata:", err);
    }
  });
