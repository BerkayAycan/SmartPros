import pandas as pd
import requests
import pdfplumber
import os
import openai
import textwrap
from dotenv import load_dotenv
from pathlib import Path
import pytesseract
from PIL import Image
import sys
import warnings
import logging

# -------------------- Uyarıları kapat --------------------
warnings.filterwarnings("ignore")
logging.getLogger("pdfplumber").setLevel(logging.CRITICAL)
sys.stderr = open(os.devnull, 'w')  # CropBox uyarılarını susturur

# -------------------- OCR yolu --------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------- .env Yolu ve API --------------------
env_path = Path(__file__).parents[1] / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("OPENAI_API_KEY .env dosyasından okunamadı!")

client = openai.OpenAI(api_key=api_key)

# -------------------- CSV'yi oku --------------------
df = pd.read_csv(r"C:\Users\Berkay\Desktop\bitirme projesi\proje\SmartPros\ai\data\DrugsData_cleaned.csv")

drug_name = input("Lütfen özetlemek istediğiniz ilacın adını girin: ")

pdf_url = df[df['ilaç adı'].str.contains(drug_name, case=False, na=False)]['küb PDF'].values[0]
print(f"\nSeçilen ilacın PDF linki: {pdf_url}")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

response = requests.get(pdf_url, headers=headers, timeout=10)

if response.status_code == 200:
    with open("temp_drug.pdf", "wb") as f:
        f.write(response.content)
    print("\nProspektüs indirildi. İşleniyor... Lütfen bekleyin.\n")
else:
    print(f"PDF indirilemedi! Sunucu kodu: {response.status_code}")
    exit()

# -------------------- PDF'den metni oku --------------------
with pdfplumber.open("temp_drug.pdf") as pdf:
    pages = pdf.pages
    all_text = ""

    for idx, page in enumerate(pages):
        page_text = page.extract_text()
        if page_text:
            all_text += page_text + "\n"
        else:
            pil_image = page.to_image(resolution=300).original
            ocr_text = pytesseract.image_to_string(pil_image, lang="tur")
            all_text += ocr_text + "\n"

os.remove("temp_drug.pdf")

if not all_text.strip():
    print("PDF'den metin okunamadı.")
    exit()

# -------------------- Metni parçalara böl --------------------
chunk_size = 3000
text_chunks = textwrap.wrap(all_text, width=chunk_size)

# -------------------- Özetleme --------------------
print("Özet hazırlanıyor... Lütfen bekleyin.\n")

summaries = []

for chunk in text_chunks:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content":
             "Sen bir ilaç prospektüsü özetleyicisisin. Yalnızca şu bilgileri **madde madde** listele:\n"
             "- Kullanım amacı\n"
             "- Kullanım dozu ve sıklığı (varsa)\n"
             "- Alerjen içerikler (varsa)\n"
             "- Önemli yan etkiler ve tehlikeler\n"
             "- Hamilelik/emzirme uyarıları\n"
             "- Kritik uyarılar\n\n"
             "**Kesinlikle** aşağıdaki bilgileri dahil etme: üretici, ruhsat sahibi, tarih, ruhsat numarası, kullanım talimatı dışındaki bilgiler.\n"
             "Madde başına maksimum 20 kelime kullan. Gereksiz detayları çıkar."},
            {"role": "user", "content": f"Şu metni özetle:\n\n{chunk}"}
        ],
        temperature=0,
        max_tokens=500
    )

    summary = response.choices[0].message.content
    summaries.append(summary)

final_summary = "\n\n".join(summaries)

print("\n✅ Özetlenen Prospektüs:\n")
print(final_summary)
