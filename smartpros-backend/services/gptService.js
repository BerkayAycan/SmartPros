const OpenAI = require('openai');
const axios = require('axios');
const pdfParse = require('pdf-parse');
const Drug = require('../models/Drug');

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function getPersonalizedSummary(userData, drugInfo) {
  const drug = await Drug.findOne({
    name: { $regex: drugInfo, $options: 'i' }
  });

  if (!drug) throw new Error("İlaç bulunamadı.");

  const pdfRes = await axios.get(drug.pdfUrl, { responseType: 'arraybuffer' });
  const pdfData = await pdfParse(pdfRes.data);
  const text = pdfData.text;

  const userInfo = `
Kullanıcı:
- Yaş: ${userData.age}
- Cinsiyet: ${userData.gender}
- Alerjiler: ${userData.allergies.join(", ")}
- Hastalıklar: ${userData.diseases.join(", ")}
- Hamilelik: ${userData.pregnancyStatus ? "Evet" : "Hayır"}
- Araç kullanımı: ${userData.drivingStatus ? "Evet" : "Hayır"}
`;

  const prompt = `${userInfo}\n\nProspektüs:\n${text.slice(0, 3000)}\n\nBu kullanıcıya uygun şekilde özetle: kullanım amacı, dozaj, alerjenler, yan etkiler, hamilelik uyarısı, kritik uyarılar.`;

  const completion = await openai.chat.completions.create({
    model: "gpt-3.5-turbo-1106",
    messages: [
      { role: "system", content: "Sen kişisel sağlık uyarılarını dikkate alarak prospektüs özetleyen bir asistansın." },
      { role: "user", content: prompt }
    ],
    max_tokens: 500,
    temperature: 0
  });

  return completion.choices[0].message.content;
}

module.exports = { getPersonalizedSummary };
