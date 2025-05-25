from fastapi.responses import JSONResponse
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
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
import warnings
import logging
import sys
import re

warnings.filterwarnings("ignore")
logging.getLogger("pdfplumber").setLevel(logging.CRITICAL)
sys.stderr = open(os.devnull, 'w')

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

env_path = Path(__file__).parents[1] / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

csv_path = Path(__file__).parents[1] / "data" / "DrugsData_cleaned.csv"
df = pd.read_csv(csv_path)

def extract_pdf_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
            else:
                image = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(image, lang="tur")
                all_text += ocr_text + "\n"
    return all_text.strip()

@app.get("/summary")
def get_summary(
    drug_name: str = Query(...),
    gender: str = Query(None),
    age: int = Query(None),
    weight: float = Query(None),
    allergies: str = Query(None),
    conditions: str = Query(None),
    current_drugs: str = Query(None),
    pregnant: bool = Query(None),
    driving: bool = Query(None)
):
    match = df[df["ilaç adı"].str.contains(drug_name, case=False, na=False)]
    if match.empty:
        return {"error": "İlaç bulunamadı."}

    pdf_url = match.iloc[0]["küb PDF"]
    response = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    if response.status_code != 200:
        return {"error": "PDF indirilemedi."}

    with open("temp_drug.pdf", "wb") as f:
        f.write(response.content)

    all_text = extract_pdf_text("temp_drug.pdf")
    os.remove("temp_drug.pdf")

    if not all_text:
        return {"error": "PDF içeriği boş veya okunamadı."}

    chunks = textwrap.wrap(all_text, width=2500)
    collected_summaries = {}

    for chunk in chunks:
        prompt = (
            "🔬 Sen bir sağlık asistanısın. Aşağıdaki prospektüs metnini özetle. Her özet bloğu, ilacın 'Kullanım Amacı' başlığına göre gruplanmalı.\n"
            "📌 Özette aşağıdaki başlıklar yer almalı (her biri için mantıklı bir web emojisiyle başla):\n"
            "💊 Kullanım amacı\n⏱️ Doz ve sıklık\n🧪 Alerjen içerikler\n🤕 Yan etkiler\n⚠️ Kritik uyarılar\n\n"
            "❗❗❗ Dikkat:\n"
            "- Ana kullanım amacı her zaman 'Genel Bilgi' başlığında yer almalı.\n"
            "- Yan kullanım amaçları başka başlıklarda gruplanmalı.\n"
            "- İçerik sade, anlaşılır ve kısa paragraflarla verilmeli.\n"
            "- Teknik terimler azaltılmalı, herkesin anlayabileceği Türkçe kullanılmalı.\n"
            "- Gereksiz tekrarlar, boş başlıklar ya da 'belirtilmemiş' gibi ifadeler kullanılmamalı.\n"
            "- Emoji destekli, okunabilirliği yüksek liste formatı tercih edilmeli.\n"
            "- Erkek kullanıcılar için hamilelik/emzirme bölümleri çıkartılmalı.\n"
            "- Kullanıcı profili:\n"
            f"👤 Cinsiyet: {gender or 'belirtilmemiş'}\n"
            f"🎂 Yaş: {age or 'belirtilmemiş'}\n⚖️ Kilo: {weight or 'belirtilmemiş'}\n"
            f"🌸 Alerjiler: {allergies or 'yok'}\n🩺 Hastalıklar: {conditions or 'yok'}\n💊 Diğer ilaçlar: {current_drugs or 'yok'}\n"
            f"🤰 Hamile mi?: {'Evet' if pregnant else 'Hayır'}\n🚗 Araç kullanıyor mu?: {'Evet' if driving else 'Hayır'}\n\n"
            f"🧾 Metin:\n{chunk}"
        )

        res = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800
        )
        content = res.choices[0].message.content

        usage_blocks = re.split(r"- Kullanım amacı\s*:", content)
        if len(usage_blocks) > 1:
            for block in usage_blocks[1:]:
                lines = block.strip().split("\n")
                title = lines[0].strip()
                summary = "- " + "\n- ".join([
                    line.strip(" -") for line in lines[1:]
                    if line.strip()
                ])
                if gender == "male":
                    summary = re.sub(r"-? ?Hamilelik.*?(\n|$)", "", summary, flags=re.IGNORECASE)
                if title not in collected_summaries:
                    collected_summaries[title] = summary
        elif content.strip():
            collected_summaries["Genel Bilgi"] = content.strip()

    if age and weight and collected_summaries:
        dose = round(weight * 0.5)
        last_key = list(collected_summaries.keys())[-1]
        collected_summaries[last_key] += f"\n\n💡 Tahmini kullanım dozu: {dose} mL/gün (doktor onayı gereklidir)."

    if not collected_summaries:
        return {"error": "Özet oluşturulamadı."}

    return JSONResponse(
        content={
            "drugName": drug_name,
            "summaries": collected_summaries,
            "purposes": list(collected_summaries.keys())
        },
        media_type="application/json; charset=utf-8"
    )

@app.get("/")
def root():
    return {"status": "FastAPI ayakta"}
