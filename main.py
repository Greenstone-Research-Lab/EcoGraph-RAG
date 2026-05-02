import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. Dosyanın kesin yolunu (absolute path) buluyoruz
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
env_path = os.path.join(current_dir, '.env')

# override=True ile daha önce RAM'de kalan boş değerleri eziyoruz
load_dotenv(dotenv_path=env_path, override=True)

# 2. Değeri okuyup tam olarak ne gördüğüne bakıyoruz
api_key = os.getenv("OPENAI_API_KEY")

# Konsola sadece API anahtarının ilk 5 karakterini yazdırarak güvenli test yapıyoruz
if api_key:
    print(f"🔍 Okunan anahtarın başı: '{api_key[:5]}...'")
    print(f"🔍 Okunan anahtarın uzunluğu: {len(api_key)} karakter")
else:
    print("❌ api_key değişkeni hala None (boş) dönüyor.")

if not api_key or api_key == "senin_openai_api_anahtarin_buraya_gelecek":
    print("⚠️ UYARI: .env dosyası okunamadı veya içindeki değer değiştirilmedi.")
else:
    print("✅ BAŞARILI: API anahtarı sisteme yüklendi.")
    
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) 
        print("⏳ LLM'e bağlanılıyor, lütfen bekleyin...")
        cevap = llm.invoke("Merhaba, test yapıyoruz. Bana sadece tek kelime ile 'Sistem Aktif' yaz.")
        print(f"🤖 LLM Yanıtı: {cevap.content}")
        
    except Exception as e:
        print(f"❌ LLM'e bağlanırken bir hata oluştu. Hata detayı:\n{e}")