# main.py
import tkinter as tk
from ReasonClassifier import ReasonClassifier
import os
from tkinter import filedialog, messagebox, Toplevel, scrolledtext, HORIZONTAL, VERTICAL
from tkinter import ttk
import pandas as pd
# Model dosyalarının bulunduğu dizinler
from utils import get_absolute_path

reason_model_dirs = {
    "Ürünüm beklediğim kalitede değildi": get_absolute_path("Urunum_bekledigim_kalitede_degil/hata_bert_finetuned_model/openvino_model_quantized"),
    "Ürünün kutusu / ambalajı sorunluydu": get_absolute_path("Urunun_kutusu_ambalaji_sorunluydu/hata_bert_finetuned_model/openvino_model_quantized"),
    "Kargo firması / çalışanı / teslimat deneyiminden memnun değilim": get_absolute_path("Kargo_firmasi_calisani_teslimat_deneyiminden_memnun_degilim/hata_bert_finetuned_model/openvino_model_quantized"),
    "Mağaza çalışanı benimle ilgilenmedi / yeterli bilgiye sahip değildi": get_absolute_path("Magaza_calisani_benimle_ilgilenmedi_yeterli_bilgiye_sahip_degildi/hata_bert_finetuned_model/openvino_model_quantized"),
    "Kampanyalar ve kupon kodları konusunda sorun yaşadım": get_absolute_path("Kampanyalar_ve_kupon_kodlari_konusunda_sorun_yasadim/hata_bert_finetuned_model/openvino_model_quantized"),
    "Kasada uzun süre bekledim / tüm kasalar açık değildi": get_absolute_path("Kasada_uzun_sure_bekledim_tum_kasalar_acik_degildi/hata_bert_finetuned_model/openvino_model_quantized"),
    "Mağaza düzensiz / karışık / kirliydi": get_absolute_path("Magaza_duzensiz_karisik_kirliydi/hata_bert_finetuned_model/openvino_model_quantized"),
    "Ödeme işlemi sırasında sorun yaşadım": get_absolute_path("Odeme_islemi_sirasinda_sorun_yasadim/hata_bert_finetuned_model/openvino_model_quantized"),
    "Ürün / marka çeşidi azdı": get_absolute_path("Urun_marka_çeşidi_azdı/hata_bert_finetuned_model/openvino_model_quantized"),
    "Ürünüm geç teslim edildi": get_absolute_path("Urunum_gec_teslim_edildi/hata_bert_finetuned_model/openvino_model_quantized"),
    "Ürünüm yanlış / eksik gönderildi": get_absolute_path("Urunum_yanlis_eksik_gonderildi/hata_bert_finetuned_model/openvino_model_quantized")
}

subreason_model_dirs = {
    "Ürünüm beklediğim kalitede değildi": get_absolute_path("Urunum_bekledigim_kalitede_degil/alt_bert_finetuned_model/openvino_model_quantized"),
    "Ürünün kutusu / ambalajı sorunluydu": get_absolute_path("Urunun_kutusu_ambalaji_sorunluydu/alt_bert_finetuned_model/openvino_model_quantized"),
    "Kargo firması / çalışanı / teslimat deneyiminden memnun değilim": get_absolute_path("Kargo_firmasi_calisani_teslimat_deneyiminden_memnun_degilim/alt_bert_finetuned_model/openvino_model_quantized"),
    "Mağaza çalışanı benimle ilgilenmedi / yeterli bilgiye sahip değildi": get_absolute_path("Magaza_calisani_benimle_ilgilenmedi_yeterli_bilgiye_sahip_degildi/alt_bert_finetuned_model/openvino_model_quantized"),
    "Kampanyalar ve kupon kodları konusunda sorun yaşadım": get_absolute_path("Kampanyalar_ve_kupon_kodlari_konusunda_sorun_yasadim/alt_bert_finetuned_model/openvino_model_quantized"),
    "Kasada uzun süre bekledim / tüm kasalar açık değildi": get_absolute_path("Kasada_uzun_sure_bekledim_tum_kasalar_acik_degildi/alt_bert_finetuned_model/openvino_model_quantized"),
    "Mağaza düzensiz / karışık / kirliydi": get_absolute_path("Magaza_duzensiz_karisik_kirliydi/alt_bert_finetuned_model/openvino_model_quantized"),
    "Ödeme işlemi sırasında sorun yaşadım": get_absolute_path("Odeme_islemi_sirasinda_sorun_yasadim/alt_bert_finetuned_model/openvino_model_quantized"),
    "Ürün / marka çeşidi azdı": get_absolute_path("Urun_marka_çeşidi_azdı/alt_bert_finetuned_model/openvino_model_quantized"),
    "Ürünüm geç teslim edildi": get_absolute_path("Urunum_gec_teslim_edildi/alt_bert_finetuned_model/openvino_model_quantized"),
    "Ürünüm yanlış / eksik gönderildi": get_absolute_path("Urunum_yanlis_eksik_gonderildi/alt_bert_finetuned_model/openvino_model_quantized")
}

tokenizer_dirs = {
    "Ürünüm beklediğim kalitede değildi": get_absolute_path("Urunum_bekledigim_kalitede_degil/hata_bert_finetuned_model/tokenizer"),
    "Ürünün kutusu / ambalajı sorunluydu": get_absolute_path("Urunun_kutusu_ambalaji_sorunluydu/hata_bert_finetuned_model/tokenizer"),
    "Kargo firması / çalışanı / teslimat deneyiminden memnun değilim": get_absolute_path("Kargo_firmasi_calisani_teslimat_deneyiminden_memnun_degilim/hata_bert_finetuned_model/tokenizer"),
    "Mağaza çalışanı benimle ilgilenmedi / yeterli bilgiye sahip değildi": get_absolute_path("Magaza_calisani_benimle_ilgilenmedi_yeterli_bilgiye_sahip_degildi/hata_bert_finetuned_model/tokenizer"),
    "Kampanyalar ve kupon kodları konusunda sorun yaşadım": get_absolute_path("Kampanyalar_ve_kupon_kodlari_konusunda_sorun_yasadim/hata_bert_finetuned_model/tokenizer"),
    "Kasada uzun süre bekledim / tüm kasalar açık değildi": get_absolute_path("Kasada_uzun_sure_bekledim_tum_kasalar_acik_degildi/hata_bert_finetuned_model/tokenizer"),
    "Mağaza düzensiz / karışık / kirliydi": get_absolute_path("Magaza_duzensiz_karisik_kirliydi/hata_bert_finetuned_model/tokenizer"),
    "Ödeme işlemi sırasında sorun yaşadım": get_absolute_path("Odeme_islemi_sirasinda_sorun_yasadim/hata_bert_finetuned_model/tokenizer"),
    "Ürün / marka çeşidi azdı": get_absolute_path("Urun_marka_çeşidi_azdı/hata_bert_finetuned_model/tokenizer"),
    "Ürünüm geç teslim edildi": get_absolute_path("Urunum_gec_teslim_edildi/hata_bert_finetuned_model/tokenizer"),
    "Ürünüm yanlış / eksik gönderildi": get_absolute_path("Urunum_yanlis_eksik_gonderildi/hata_bert_finetuned_model/tokenizer")
}


# ReasonClassifier sınıfını oluştur
classifier = ReasonClassifier(reason_model_dirs, subreason_model_dirs,tokenizer_dirs)

# Uygulama durumu için global değişken
selected_file_path = None



def select_file():
    global selected_file_path
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        selected_file_path = file_path
        df = pd.read_excel(selected_file_path)

        reason_counts = df['Müşteri Reasonı'].value_counts()

        dist_window = Toplevel()
        dist_window.title("Reason Dağılımı")

        # Pencere boyutlandırması: geniş ve yüksek, içeriğe daha çok alan sağlar
        dist_window.geometry('700x600')  # Genişlik x Yükseklik (px cinsinden)

        text_widget = scrolledtext.ScrolledText(
            dist_window,
            wrap=tk.NONE,
            width=80,   # daha geniş
            height=30,  # daha yüksek
            borderwidth=0,
            highlightthickness=0
        )
        text_widget.pack(padx=10, pady=10, fill='both', expand=True)

        xscrollbar = tk.Scrollbar(dist_window, orient=HORIZONTAL, command=text_widget.xview)
        xscrollbar.pack(side='bottom', fill='x')
        text_widget.configure(xscrollcommand=xscrollbar.set)

        text_widget.tag_configure('header', font=('Segoe UI', 16, 'bold'), foreground='#003366', spacing3=10)
        text_widget.tag_configure('bold', font=('Segoe UI', 14, 'bold'))
        text_widget.tag_configure('normal', font=('Segoe UI', 14), spacing1=2, spacing3=4)

        text_widget.insert(tk.END, "Okunan Verideki Müşteri Reasonı Dağılımı:\n\n", 'header')

        for reason, count in reason_counts.items():
            text_widget.insert(tk.END, "• ", 'normal')
            text_widget.insert(tk.END, reason, 'bold')
            text_widget.insert(tk.END, f": {count}\n", 'normal')

        text_widget.config(state='disabled')
    else:
        messagebox.showwarning("Uyarı", "Herhangi bir dosya seçilmedi.")


def run_prediction():
    global selected_file_path
    if not selected_file_path:
        messagebox.showwarning("Uyarı", "Lütfen önce bir Excel dosyası seçin.")
        return
    try:
        classifier.process_excel(selected_file_path)
        messagebox.showinfo("Başarılı", f"Tahminler tamamlandı. Çıktı dosyası: {classifier.output_path}")
    except Exception as e:
        messagebox.showerror("Hata", f"Tahmin yapılırken bir hata oluştu:\n{str(e)}")



def main():
    window = tk.Tk()
    window.title("Reason & Subreason Tahmin Aracı")
    window.geometry("500x320")
    window.config(bg="#f4f4f9")
    window.resizable(False, False)

    style = ttk.Style()
    style.theme_use("clam")

    style.configure("Green.TButton",
                    background="#388E3C",
                    foreground="white",
                    font=("Arial", 12),
                    padding=10)
    style.map("Green.TButton",
              background=[("active", "#2E7D32")],
              foreground=[("active", "white")])

    style.configure("Blue.TButton",
                    background="#1976D2",
                    foreground="white",
                    font=("Arial", 12),
                    padding=10)
    style.map("Blue.TButton",
              background=[("active", "#1565C0")],
              foreground=[("active", "white")])

    header_label = tk.Label(window, text="Reason & Subreason Tahmin Aracı",
                            font=("Arial", 18, "bold"), fg="#2c3e50", bg="#f4f4f9")
    header_label.pack(pady=20)

    select_button = ttk.Button(window, text="Excel Dosyası Seç",
                               command=select_file, style="Green.TButton", width=20)
    select_button.pack(pady=10)

    predict_button = ttk.Button(window, text="Tahmin Yap",
                                command=run_prediction, style="Blue.TButton", width=20)
    predict_button.pack(pady=10)

    global status_label
    status_label = tk.Label(window, text="Bir dosya seçip tahmin yapın.",
                            font=("Arial", 10), fg="#7f8c8d", bg="#f4f4f9")
    status_label.pack(pady=20)

    window.mainloop()


if __name__ == "__main__":
    main()
