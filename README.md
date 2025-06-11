# 💊 SmartPros – Akıllı İlaç Bilgi Sistemi

SmartPros, ilaç prospektüslerini kullanıcı dostu bir şekilde sunan, özelleştirilmiş özetler oluşturan ve kritik ilaç bilgilerini ön plana çıkaran Flutter tabanlı bir mobil sağlık uygulamasıdır. Flask destekli backend sistemiyle çalışır.

## 🚀 Özellikler

- 🔍 **İlaç Arama & Otomatik Tamamlama**  
  Kullanıcılar sadece birkaç harf yazarak ilaçları kolayca arayabilir ve hızlıca seçebilir.

- 📄 **Akıllı Prospektüs Özeti**  
  Prospektüs bilgileri yapay zeka desteğiyle özetlenir, **doz**, **yan etki**, **kullanım talimatı** gibi kritik başlıklar ön plana çıkarılır.

- 🕒 **Özet Geçmişi**  
  Önceki ilaç aramaları ve oluşturulan özetler kullanıcı geçmişinde tutulur.

- 🌐 **Web API Entegrasyonu**  
  Flask altyapılı backend sistemi üzerinden veri alışverişi yapılır.

## 🛠️ Kullanılan Teknolojiler

- Flutter (Frontend)
- Python + Flask (Backend)
- Uvicorn (Web servis çalıştırma)
- OpenAI (Yapay Zeka ile metin özetleme)
- Dart / Pub Paketleri

## 🧪 Kurulum ve Çalıştırma

### Flutter Uygulaması

```bash
git clone https://github.com/kullanici-adi/SmartPros.git
cd SmartPros
flutter pub get
flutter run
