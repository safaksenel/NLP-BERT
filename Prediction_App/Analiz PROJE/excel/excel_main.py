import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
from pathlib import Path
input_file_path = ""
output_file_path = ""

def select_input_file():
    global input_file_path
    input_file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    input_label.config(
        text=f"Girdi Dosyası: {Path(input_file_path).name}" if input_file_path else "Girdi Dosyası seçilmedi")
    root.update()

def select_output_file():
    global output_file_path
    output_file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    output_label.config(
        text=f"Çıktı Dosyası: {Path(output_file_path).name}" if output_file_path else "Çıktı Dosyası seçilmedi")
    root.update()

def compare_and_save():
    if not input_file_path or not output_file_path:
        messagebox.showwarning("Uyarı", "Lütfen hem input hem de output dosyalarını seçin.")
        return
    try:
        # Dosyaları oku
        df = pd.read_excel(input_file_path)
        results_df = pd.read_excel(output_file_path)
        # "MAP" ve "Müşteri Reasonı" sütunlarının mevcut olduğunu kontrol et
        if "Müşteri Reasonı" not in df.columns:
            raise ValueError("Excel dosyasında 'Müşteri Reasonı'  sütunları eksik.")
        if  "MAP" not in df.columns:
            raise ValueError("Excel dosyasında   'MAP' sütunları eksik.")
        # Doğru ve yanlışları ayırma
        correct_df = results_df[results_df['MAP'] == df['MAP']]
        incorrect_df = results_df[results_df['MAP'] != df['MAP']]
        # Yanlışların doğru MAP değerini ekleyelim
        incorrect_df['Gerçek MAP'] = df['MAP']
        # Excel dosyalarına kaydet
        correct_filename = f"{Path(input_file_path).stem}_dogrular.xlsx"
        incorrect_filename = f"{Path(input_file_path).stem}_yanlislar.xlsx"
        correct_df.to_excel(correct_filename, index=False)
        incorrect_df.to_excel(incorrect_filename, index=False)
        # Kullanıcıyı bilgilendir
        messagebox.showinfo("Başarılı",
                            f"Doğrular ve Yanlışlar dosyaları kaydedildi.\n\nDoğrular: {correct_filename}\nYanlışlar: {incorrect_filename}")
    except Exception as e:
        messagebox.showerror("Hata", f"Bir hata oluştu: {e}")

# GUI Başlat
root = tk.Tk()
root.title("Reason ve Subreason Karşılaştırma Aracı")
root.geometry("750x520")
root.resizable(False, False)
root.configure(bg="#f5f5f5")
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Segoe UI", 11, "bold"),
                foreground="#ffffff", background="#4CAF50", padding=10)
style.map("TButton", background=[("active", "#388E3C")])
style.configure("TLabel", font=("Segoe UI", 10), background="#f5f5f5", foreground="#333333")
# Başlık
header = ttk.Label(root, text="📊 Reason ve Subreason Kıyaslama Aracı", font=("Segoe UI", 16, "bold"),
                   foreground="#4CAF50")
header.pack(pady=30)
# Butonlar ve etiketler
btn1 = ttk.Button(root, text="1️⃣    Girdi Verilerini Seç", command=select_input_file)
btn1.pack(pady=10)
input_label = ttk.Label(root, text="Girdi Dosyası seçilmedi", font=("Segoe UI", 11))
input_label.pack()
btn2 = ttk.Button(root, text="2️⃣    Çıktı Verilerini Seç", command=select_output_file)
btn2.pack(pady=10)
output_label = ttk.Label(root, text="Çıktı Dosyası seçilmedi", font=("Segoe UI", 11))
output_label.pack()
btn3 = ttk.Button(root, text="3️⃣    Doğru ve Yanlışları Karşılaştır ve Kaydet", command=compare_and_save)
btn3.pack(pady=30)
root.mainloop()