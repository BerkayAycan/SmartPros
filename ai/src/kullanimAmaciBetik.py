import requests
from bs4 import BeautifulSoup
import time

# 100 ilacın listesi
DRUGS = [
    "XANAX", "PROZAC", "LOXAPINE", "RIVOTRIL", "LARGACTIL", "SERTRALINE", "FLUOXETINE",
    "ESCITALOPRAM", "CIPRALEX", "VALIUM", "ATIVAN", "DIAZEPAM", "LORAZEPAM", "OLANZAPINE",
    "RISPERIDONE", "SEROQUEL", "ABILIFY", "TRAZODONE", "HALOPERIDOL", "ZYPREXA", "LEXAPRO",
    "VENLAFAXINE", "EFFEXOR", "BUPROPION", "WELLBUTRIN", "PAROXETINE", "PAXIL", "ZOLOFT",
    "LITHIUM", "CARBAMAZEPINE", "TEGRETOL", "VALPROATE", "DEPAKINE", "LAMOTRIGINE", "LAMICTAL",
    "MIRTAZAPINE", "REMERON", "CLOZAPINE", "CLOPIXOL", "MODAFINIL", "RITALIN", "CONCERTA",
    "METHYLPHENIDATE", "AMPHETAMINE", "ADDERALL", "PHENOBARBITAL", "PHENYTOIN", "TOPIRAMATE",
    "KEPPRA", "LEVETIRACETAM", "GABAPENTIN", "PREGABALIN", "LYRICA", "NALTREXONE", "SUBOXONE",
    "METHADONE", "BUPRENORPHINE", "CLONAZEPAM", "TAMOXIFEN", "LETROZOLE", "ANASTROZOLE",
    "FINASTERIDE", "DUTASTERIDE", "TAMSULOSIN", "ALFUZOSIN", "SILDENAFIL", "TADALAFIL",
    "ASPIRIN", "PARACETAMOL", "IBUPROFEN", "NAPROXEN", "DICLOFENAC", "MELOXICAM",
    "KETOPROFEN", "TRAMADOL", "CODEINE", "MORPHINE", "FENTANYL", "OXYCODONE", "HYDROCODONE",
    "NALOXONE", "GLIBENCLAMIDE", "GLIPIZIDE", "METFORMIN", "INSULIN", "GLIMEPIRIDE",
    "SITAGLIPTIN", "LINAGLIPTIN", "PIOGLITAZONE", "ROSIGLITAZONE", "ATORVASTATIN",
    "SIMVASTATIN", "ROSUVASTATIN", "EZETIMIBE", "RAMIPRIL", "LISINOPRIL", "LOSARTAN", "VALSARTAN"
]

BASE_URL = "https://www.google.com/search?q={query}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
}

def get_usage_purpose(drug):
    query = f"{drug} ne için kullanılır"
    url = BASE_URL.format(query=query.replace(" ", "+"))
    try:
        resp = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(resp.text, "html.parser")
        desc = soup.find("div", class_="BNeawe s3v9rd AP7Wnd")
        if desc:
            return desc.text.split(". ")[0]  # İlk cümleyi al
    except Exception as e:
        print(f"Hata ({drug}):", e)
    return "Kullanım amacı bulunamadı."

override_info = {}

for i, drug in enumerate(DRUGS, 1):
    print(f"[{i}/{len(DRUGS)}] {drug} için kullanım amacı aranıyor...")
    purpose = get_usage_purpose(drug)
    override_info[drug] = purpose
    time.sleep(2)  # Google'dan engel yememek için bekleme

# Python dosyasına yaz
with open("override_drug_info.py", "w", encoding="utf-8") as f:
    f.write("override_info = {\n")
    for drug, purpose in override_info.items():
        f.write(f'    "{drug}": "{purpose}",\n')
    f.write("}\n")

print("override_drug_info.py başarıyla oluşturuldu.")
