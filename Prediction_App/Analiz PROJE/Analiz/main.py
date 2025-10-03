import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os

input_file_path = ""
output_file_path = ""

def select_input_file():
    global input_file_path
    input_file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    filename = os.path.basename(input_file_path) if input_file_path else "Girdi Dosyası seçilmedi"
    input_label.config(text=f"Girdi Dosyası: {filename}")

def select_output_file():
    global output_file_path
    output_file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    filename = os.path.basename(output_file_path) if output_file_path else "Çıktı Dosyası seçilmedi"
    output_label.config(text=f"Çıktı Dosyası: {filename}")

def compare_predictions():
    if not input_file_path or not output_file_path:
        messagebox.showwarning("Uyarı", "Lütfen hem input hem de output dosyalarını seçin.")
        return

    try:
        df_true = pd.read_excel(input_file_path)
        df_pred = pd.read_excel(output_file_path)
    except Exception as e:
        messagebox.showerror("Hata", f"Excel dosyası okunamadı:\n{e}")
        return

    if df_true.shape[0] != df_pred.shape[0]:
        messagebox.showerror("Hata", "Girdi ve çıktı dosyaları aynı uzunlukta olmalıdır.")
        return

    dogru_sayisi = 0
    yanlis_sayisi = 0
    reason_analiz = {r: {'dogru': 0, 'yanlis': 0} for r in df_true['Müşteri Reasonı'].unique()}
    subreason_analiz = {}

    for idx, row in df_pred.iterrows():
        true_reason = df_true.at[idx, 'Müşteri Reasonı']
        true_subreason = df_true.at[idx, 'MAP']
        predicted_subreason = row['MAP']

        if predicted_subreason == true_subreason:
            dogru_sayisi += 1
            reason_analiz[true_reason]['dogru'] += 1
        else:
            yanlis_sayisi += 1
            reason_analiz[true_reason]['yanlis'] += 1

        if true_reason not in subreason_analiz:
            subreason_analiz[true_reason] = {}
        if true_subreason not in subreason_analiz[true_reason]:
            subreason_analiz[true_reason][true_subreason] = {'dogru': 0, 'yanlis': 0}
        if predicted_subreason == true_subreason:
            subreason_analiz[true_reason][true_subreason]['dogru'] += 1
        else:
            subreason_analiz[true_reason][true_subreason]['yanlis'] += 1

    toplam_data = len(df_pred)
    dogru_oran = (dogru_sayisi / toplam_data) * 100

    output_lines = [
        "📊 ANALİZ SONUÇLARI",
        f"Toplam Yorum: {toplam_data}",
        f"Genel Doğruluk: %{dogru_oran:.2f} (✅ Doğru: {dogru_sayisi}, ❌ Yanlış: {yanlis_sayisi})",
        "\n🧩 Ana Reason Doğruluk Analizi:"
    ]

    for reason, data in reason_analiz.items():
        total = data['dogru'] + data['yanlis']
        accuracy = (data['dogru'] / total) * 100 if total > 0 else 0
        output_lines.append(
            f"{reason}: Toplam: {total}, ✅ {data['dogru']}, ❌ {data['yanlis']}, 🎯 Doğruluk: %{accuracy:.2f}"
        )

    for reason, sub_dict in subreason_analiz.items():
        output_lines.append(f"\n{reason} için Alt Reason Analizi:")
        for subreason, stats in sub_dict.items():
            total = stats['dogru'] + stats['yanlis']
            accuracy = (stats['dogru'] / total) * 100 if total > 0 else 0
            output_lines.append(
                f"\t{subreason}: Toplam: {total}, ✅ {stats['dogru']}, ❌ {stats['yanlis']}, 🎯 %{accuracy:.2f}"
            )

    show_result_window('\n'.join(output_lines))

def show_result_window(text_content):
    result_window = tk.Toplevel()
    result_window.title("Analiz Sonuçları")
    result_window.geometry("950x600")
    result_window.configure(bg="#ffffff")

    text = tk.Text(result_window, wrap='word', font=("Consolas", 10), bg="#f9f9f9", fg="#202020")
    text.pack(padx=15, pady=15, fill="both", expand=True)
    text.insert('1.0', text_content)
    text.config(state='disabled')

# Ana pencere
root = tk.Tk()
root.title("Reason/Subreason Kıyaslama Aracı")
root.geometry("750x520")
root.resizable(False, False)
root.configure(bg="#f0f4f7")

# Stil ayarları
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Segoe UI", 11, "bold"), foreground="#ffffff", background="#007acc", padding=10)
style.map("TButton", background=[("active", "#005f99")])
style.configure("TLabel", font=("Segoe UI", 10), background="#f0f4f7")

# Başlık
header = ttk.Label(root, text="📊 Reason ve Subreason Tahmin Kıyaslama Aracı", font=("Segoe UI", 16, "bold"))
header.pack(pady=30)

# Butonlar ve Etiketler
btn1 = ttk.Button(root, text="1️⃣    Input Verileri Seç", command=select_input_file)
btn1.pack(pady=10)

input_label = ttk.Label(root, text="Girdi Dosyası seçilmedi")
input_label.pack()

btn2 = ttk.Button(root, text="2️⃣    Output Verileri Seç", command=select_output_file)
btn2.pack(pady=10)

output_label = ttk.Label(root, text="Çıktı Dosyası seçilmedi")
output_label.pack()

btn3 = ttk.Button(root, text="3️⃣    Kıyasla ve Analiz Et", command=compare_predictions)
btn3.pack(pady=30)

# Uygulama döngüsü
root.mainloop()
