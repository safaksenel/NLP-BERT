# ReasonClassifier.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openvino.runtime import Core
from transformers import AutoTokenizer
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from difflib import get_close_matches
from tkinter import Toplevel, Label, filedialog
from tkinter.ttk import Progressbar

from utils import get_absolute_path  # EKLENDİ

class ReasonClassifier:
    def __init__(self, reason_model_dirs, subreason_model_dirs, tokenizer_dirs):
        ie = Core()

        # Reason modellerini yükle
        self.reason_models = {}
        self.reason_tokenizers = {}
        for reason, model_dir in reason_model_dirs.items():
            model_path = get_absolute_path(f"{model_dir}/openvino_model.xml")
            print(f"Loading reason model '{reason}' from {model_path}...")
            model = ie.read_model(model_path)
            compiled_model = ie.compile_model(model, device_name="CPU")
            self.reason_models[reason] = compiled_model

            tokenizer_path = get_absolute_path(tokenizer_dirs.get(reason, ""))
            if not tokenizer_path:
                raise ValueError(f"Tokenizer path not found for reason '{reason}'")
            print(f"Loading tokenizer for reason '{reason}' from {tokenizer_path}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.reason_tokenizers[reason] = tokenizer

        # Subreason modellerini yükle
        self.subreason_models = {}
        self.subreason_tokenizers = {}
        for reason, model_dir in subreason_model_dirs.items():
            model_path = get_absolute_path(f"{model_dir}/openvino_model.xml")
            print(f"Loading subreason model '{reason}' from {model_path}...")
            model = ie.read_model(model_path)
            compiled_model = ie.compile_model(model, device_name="CPU")
            self.subreason_models[reason] = compiled_model

            tokenizer_path = get_absolute_path(tokenizer_dirs.get(reason, ""))
            if not tokenizer_path:
                raise ValueError(f"Tokenizer path not found for subreason '{reason}'")
            print(f"Loading tokenizer for subreason '{reason}' from {tokenizer_path}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.subreason_tokenizers[reason] = tokenizer


        self.subreason_labels = {
            "Ürünün kutusu / ambalajı sorunluydu": {
                0: "Farklı bir markaya ait kutu ile gönderildi",
                1: "Mağazalarda hediye paketi bulunmuyor",
                2: "Poşet ücreti alınıyor",
                3: "Ürün alarmlı gönderildi",
                4: "Ürün kutusuz gönderildi​",
                5: "Ürünler kapaksız kutu ile gönderildi",
                6: "Ürünüm alarmı çıkartılmadan paketlendi",
                7: "Ürünün kutusu hasarlıydı​",
                8: "Ürünün kutusu kirliydi / yıpranmıştı"
            },
            "Ürünüm beklediğim kalitede değildi": {
                0: "Değişim iade esnasında problem yaşadım",
                1: "Farklı model ürün gönderildi",
                2: "Kalıbı küçüktü/büyüktü",
                3: "Orijinal olduğunu düşünmüyorum",
                4: "Teslimat süreci uzun sürdü",
                5: "Ürün kullanılmıştı",
                6: "Ürün yanlış renk gönderildi",
                7: "Ürünüm defolu gönderildi",
                8: "Ürünüm kirli/lekeli gönderildi",
                9: "Ürünümün kalitesinden memnun kalmadım",
                10: "Ürünümün materyalinden memnun kalmadım",
                11: "Ürünümün orijinalliği konusunda tereddütlerim var",
                12: "Ürünün kalıp hatası vardı",
                13: "Ürünün kutusu hasarlıydı"
            },
            "Kargo firması / çalışanı / teslimat deneyiminden memnun değilim": {
                0: "Adresim dışında bir yere teslimat gerçekleştirildi",
                1: "Belirtilen sürede teslimat gerçekleştirilmedi",
                2: "Farklı model ürün gönderildi",
                3: "Kargo görevlisi kargomun şubeden alınması gerektiğini söyledi​",
                4: "Kargo temsilcisinin üslubu kabaydı",
                5: "Kargo ürünümü teslim etmeden iadeye yönlendirdi",
                6: "Siparişim parçalı teslim edildi",
                7: "Ürünüm defolu geldi",
                8: "Ürünüm teslim edilmediği halde teslim edilmiş gösterildi"
            },
            "Mağaza çalışanı benimle ilgilenmedi / yeterli bilgiye sahip değildi": {
                0: "Mağaza çalışan sayısı yeterli değildi",
                1: "Mağaza çalışanları ilgisizdi",
                2: "Mağaza çalışanı iade / değişim konusunda yardımcı olmadı",
                3: "Mağaza çalışanı yardım istememe rağmen ilgilenmedi",
                4: "Mağaza çalışanı yeterli bilgiye sahip değildi",
                5: "Mağaza çalışanı ödeme sırasında hatalı işlem yaptı",
                6: "Mağaza çalışanının üslubu / tavrı iyi değildi"
            },
            "Kampanyalar ve kupon kodları konusunda sorun yaşadım": {
                0: "Aldığım ürünlerin fişi ayrı kesildiğinden kampanyadan yararlanamadım",
                1: "Alışverişlerimde kart puanlarımı kullanamıyorum",
                2: "Değişim / iade esnasında kampanya dikkate alınmadı",
                3: "Kampanya içeriği yeterince anlaşılır değil",
                4: "Kampanya süresi dolmasına rağmen etiketler güncellenmemiş",
                5: "Kampanya çeşitleri ve kapsamı arttırılmalı",
                6: "Kampanya öncesinde ürün fiyatlarında artış gerçekleştiriliyor",
                7: "Kampanyanın geçerli olduğu ürün sayısı yetersizdi",
                8: "Mağaza çalışanı kampanya hakkında bilgilendirmedi"
            },
            "Kasada uzun süre bekledim / tüm kasalar açık değildi": {
                0: "Değişim / iade işlemlerinde çok bekledim",
                1: "Etiket fiyatı farklıydı",
                2: "Kasada yan ürün satışı için tanıtım yapılıyor",
                3: "Kasaların tamamı açık değildi",
                4: "Tek kasa açıktı",
                5: "Uzun süre bekledim / işlemler yavaş ilerliyor",
                6: "Çalışan hatalı ürünü paketledi",
                7: "Çalışan sayısı yetersizdi",
                8: "Çalışanlar farklı işlerle ilgileniyordu",
                9: "Çalışanlar ilgili / güler yüzlü değildi",
                10: "Ödeme sırasında problem yaşadım / hatalı bilgilendirme yapıldı",
                11: "Ürün üzerinde alarm unutulmuş"
            },
            "Ödeme işlemi sırasında sorun yaşadım": {
                0: "Etiket fiyatı ile kasadaki fiyat farklıydı",
                1: "Fiyatlar internet satışında daha ucuzdu",
                2: "Kampanyalar konusunda hatalı bilgilendirme yapıldı",
                3: "Kartıma tanımlı puan kullandırılmadı",
                4: "Kasa görevlisi para üstü vermek istemedi",
                5: "Kasa görevlisinin tavrından / üslubundan hoşlanmadım",
                6: "Kasada pos cihazı çalışmadı",
                7: "Kasadan satın almadığım bir ürün geçirildi",
                8: "Kredi kartına vade farkı istendi",
                9: "QR ile ödeme gerçekleştiremedim",
                10: "İstediğim miktarda taksit yapılmadı"
            },
            "Mağaza düzensiz / karışık / kirliydi": {
                0: "Ayakkabı denerken oturulacak bir yer bulunmuyordu",
                1: "Beğendiğim ayakkabının bir teki bulunamadı",
                2: "Kasa arızalandığı için karışıklık oldu",
                3: "Mağaza pazar yeri gibiydi",
                4: "Mağazada kasanın konumu yüzünden karışıklık oldu",
                5: "Mağazanın bazı bölümlerinde ürün yerleştirmeleri düzenli değildi",
                6: "Mağazanın kokusu rahatsız ediciydi",
                7: "Müşterilerin denedikleri ürünler ortada bırakılmıştı",
                8: "Raflardaki etiketler ve kasada çıkan fiyat birbirinden farklıydı"
            },
            "Ürün / marka çeşidi azdı": {
                0: "Aradığım numarayı bulamadım",
                1: "Belirli markalarda ürün çeşidi azdı",
                2: "Erkek aksesuarında çeşit azdı",
                3: "Erkek ayakkabı kategorisinde çeşit azdı",
                4: "Giyim kategorisinde yeterli ürün çeşiti yok",
                5: "Kadın aksesuarlarında çeşit azdı",
                6: "Kadın ayakkabı kategorisinde çeşit azdı",
                7: "Mağaza çalışanları yardımcı olmadığından ürünümü bulamadım",
                8: "Mağazada olan ürünler internet sitesinde yoktu",
                9: "Çocuk ürünlerinde çeşit azdı",
                10: "Üst giyim kategorisinde çeşit azdı",
                11: "İnternet sitesinde gördüğüm ürünler mağazada yoktu"
            },
            "Ürünüm geç teslim edildi": {
                0: "Değişim/İade/inceleme süreçlerinde gecikme yaşandı",
                1: "Hatalı/kusurlu ürün teslim edildi",
                2: "Kargo şirketiyle problem yaşadım",
                3: "Siparişim parçalı şekilde gönderildi",
                4: "Teslimat süreci uzun sürdü",
                5: "Ürünüm depodan geç çıktı",
                6: "Ürünün paketlemesi konusunda sorun yaşadım"
            },
            "Ürünüm yanlış / eksik gönderildi": {
                0: "Eksik ürün gönderildi",
                1: "Farklı kategoriden bir ürün gönderildi",
                2: "Farklı markaya ait farklı ürün gönderildi",
                3: "Farklı model ürün gönderildi",
                4: "Kalıbı küçüktü/büyüktü",
                5: "Müşteri temsilcisi yardımcı olmadı",
                6: "Ürün yanlış numara/beden gönderildi",
                7: "Ürün yanlış renk gönderildi",
                8: "Ürünüm defolu geldi",
                9: "Ürünümün iki teki birbirinden farklıydı/aynıydı",
                10: "Ürünümün kalitesinden memnun kalmadım"
            }

        }

    def predict_reason(self, text, true_reason):
        true_reason = str(true_reason)
        if true_reason in self.reason_models:
            self.true_reason = true_reason
            model = self.reason_models[true_reason]
            tokenizer = self.reason_tokenizers[true_reason]

            inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=256)
            inputs_ov = {k: np.array(v) for k, v in inputs.items()}
            outputs = model(inputs_ov)  # infer yerine direkt model çağrısı
            logits = list(outputs.values())[0]
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[0]

            pred = 1 if probs[1] > 0.70 else 0
            return "Doğru Reason" if pred == 0 else "Hatalı reason"

        available_reasons = list(self.reason_models.keys())
        closest_matches = get_close_matches(true_reason, available_reasons, n=1, cutoff=0.6)
        if closest_matches:
            closest_reason = closest_matches[0]
            self.true_reason = closest_reason
            model = self.reason_models[closest_reason]
            tokenizer = self.reason_tokenizers[closest_reason]

            inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=256)
            inputs_ov = {k: np.array(v) for k, v in inputs.items()}
            outputs = model(inputs_ov)  # infer yerine direkt model çağrısı
            logits = list(outputs.values())[0]
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[0]

            pred = 1 if probs[1] > 0.70 else 0
            return "Doğru Reason" if pred == 0 else "Hatalı reason"
        else:
            return "Model bulunamadı"

    def predict_subreason(self, text, true_reason):
        true_reason = self.true_reason
        if true_reason not in self.subreason_models or true_reason not in self.subreason_labels:
            return "Model veya etiketler bulunamadı"

        tokenizer = self.subreason_tokenizers[true_reason]
        model = self.subreason_models[true_reason]

        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=256)
        inputs_ov = {k: np.array(v) for k, v in inputs.items()}
        outputs = model(inputs_ov)  # infer yerine direkt model çağrısı
        logits = list(outputs.values())[0]

        predicted_id = int(np.argmax(logits, axis=1)[0])
        return self.subreason_labels[true_reason].get(predicted_id, "Hatalı")

    def process_excel(self, input_path):
        df = pd.read_excel(input_path)
        results = []
        total = len(df)

        # Progressbar için pencere aç
        progress_window = Toplevel()
        progress_window.title("İşlem Devam Ediyor")
        Label(progress_window, text="Tahmin Yapılıyor, Lütfen Bekleyin...").pack(pady=10)

        percent_label = Label(progress_window, text="0%")
        percent_label.pack()

        progress_bar = Progressbar(progress_window, length=300, mode='determinate', maximum=total)
        progress_bar.pack(pady=10)
        progress_window.update()

        # tqdm yine kullanılır ama GUI'ye yansımaz, sadece konsol içindir.
        for index, row in tqdm(df.iterrows(), total=total, desc="Yorumlar işleniyor"):
            comment = row['Yorum']
            true_reason = row['Müşteri Reasonı']

            predicted_reason = self.predict_reason(comment, true_reason)
            print(f"Tahmin Edilen Reason: {predicted_reason}")

            if predicted_reason == "Doğru Reason":
                predicted_subreason = self.predict_subreason(comment, true_reason)
                print(f" Tahmin Edilen Alt Reason: {predicted_subreason}")
            else:
                predicted_subreason = "Hatalı reason"
                print("Tahmin Edilen Alt Reason: Hatalı reason seçimi")

            results.append({
                'Müşteri Reasonı': true_reason,
                'Yorum': comment,
                'MAP': predicted_subreason
            })

            # Progress bar ve yüzde güncelle
            progress_bar['value'] = index + 1
            percent = int(((index + 1) / total) * 100)
            percent_label.config(text=f"%{percent}")
            progress_window.update()

        progress_window.destroy()

        # Kaydetme dialogu
        self.output_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                        filetypes=[("Excel Dosyaları", "*.xlsx")])
        if not self.output_path:
            print("Dosya kaydetme işlemi iptal edildi.")
            return

        results_df = pd.DataFrame(results)
        results_df.to_excel(self.output_path, index=False)
        print(f"Sonuçlar '{self.output_path}' dosyasına kaydedildi.")