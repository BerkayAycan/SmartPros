import pandas as pd
import requests
import pdfplumber
import os
import openai
import textwrap
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# OpenAI API anahtarını .env dosyasından al
openai.api_key = os.getenv("OPENAI_API_KEY")

# CSV dosyasını oku
df = pd.read_csv(r"C:\Users\Berkay\Desktop\bitirme projesi\proje\SmartPros\ai\data\DrugsData_cleaned.csv")

# Kullanıcıdan ilaç adı al
drug_name = input("Lütfen özetlemek istediğiniz ilacın adını girin: ")

# İlacın PDF linkini bul
pdf_url = df[df['ilaç adı'].str.contains(drug_name, case=False, na=False)]['küb PDF'].values[0]
print(f"Seçilen ilacın PDF linki: {pdf_url}")

# User-Agent header ile PDF'yi indir
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

response = requests.get(pdf_url, headers=headers, timeout=10)

if response.status_code == 200:
    with open("temp_drug.pdf", "wb") as f:
        f.write(response.content)
    print("PDF başarıyla indirildi.\n")
else:
    print(f"PDF indirilemedi! Sunucu kodu: {response.status_code}")
    exit()

# PDF'den metni oku
with pdfplumber.open("temp_drug.pdf") as pdf:
    pages = pdf.pages
    text = "\n".join(page.extract_text() for page in pages if page.extract_text())

# Geçici PDF dosyasını sil
os.remove("temp_drug.pdf")

# Metni parçalara böl (her parça max 3000 karakter olsun)
chunk_size = 3000
text_chunks = textwrap.wrap(text, width=chunk_size)

# Chunk'ları tek tek özetle
summaries = []

for idx, chunk in enumerate(text_chunks):
    print(f"Chunk {idx+1}/{len(text_chunks)} özetleniyor...")

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "Sen bir ilaç prospektüs özetleyicisisin. Metni anlaşılır ve kısa bir şekilde özetle."},
            {"role": "user", "content": f"Şu metni özetle:\n\n{chunk}"}
        ],
        temperature=0.3,
        max_tokens=500
    )

    summary_text = response.choices[0].message.content
    summaries.append(summary_text)

# Tüm özetleri birleştir
final_summary = "\n\n".join(summaries)

print("\n✅ Özetlenen Prospektüs:\n")
print(final_summary)
