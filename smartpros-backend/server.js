const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

// Rotalar
const summaryRoutes = require('./routes/summary');
const drugRoutes = require('./routes/drugs');
const historyRoutes = require('./routes/history'); // ⬅️ bunu yukarı aldık

app.use('/api/summary', summaryRoutes);
app.use('/api/drugs', drugRoutes);
app.use('/api/history', historyRoutes); // ⬅️ bunu da buraya aldık

// PORT
const PORT = process.env.PORT || 5000;

// MongoDB Bağlantısı
mongoose.connect(process.env.MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
  .then(() => {
    console.log('✅ MongoDB Bağlandı');
    app.listen(PORT, () => console.log(`🚀 Sunucu ${PORT} portunda çalışıyor`));
  })
  .catch(err => {
    console.error('❌ MongoDB bağlantı hatası:', err);
  });
