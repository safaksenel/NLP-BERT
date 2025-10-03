import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.intel.openvino import OVQuantizer, OVConfig
from neural_compressor.config import PostTrainingQuantConfig

# Reason modelleri dizini (ana modeller)
reason_model_dirs = {
    "Ürünüm beklediğim kalitede değildi": "Urunum_bekledigim_kalitede_degil/hata_bert_finetuned_model",
    "Ürünün kutusu / ambalajı sorunluydu": "Urunun_kutusu_ambalaji_sorunluydu/hata_bert_finetuned_model",
    "Kargo firması / çalışanı / teslimat deneyiminden memnun değilim": "Kargo_firmasi_calisani_teslimat_deneyiminden_memnun_degilim/hata_bert_finetuned_model",
    "Mağaza çalışanı benimle ilgilenmedi / yeterli bilgiye sahip değildi": "Magaza_calisani_benimle_ilgilenmedi_yeterli_bilgiye_sahip_degildi/hata_bert_finetuned_model",
    "Kampanyalar ve kupon kodları konusunda sorun yaşadım": "Kampanyalar_ve_kupon_kodlari_konusunda_sorun_yasadim/hata_bert_finetuned_model",
    "Kasada uzun süre bekledim / tüm kasalar açık değildi": "Kasada_uzun_sure_bekledim_tum_kasalar_acik_degildi/hata_bert_finetuned_model",
    "Mağaza düzensiz / karışık / kirliydi": "Magaza_duzensiz_karisik_kirliydi/hata_bert_finetuned_model",
    "Ödeme işlemi sırasında sorun yaşadım": "Odeme_islemi_sirasinda_sorun_yasadim/hata_bert_finetuned_model",
    "Ürün / marka çeşidi azdı":"Urun_marka_çeşidi_azdı/hata_bert_finetuned_model",
    "Ürünüm geç teslim edildi":"Urunum_gec_teslim_edildi/hata_bert_finetuned_model",
    "Ürünüm yanlış / eksik gönderildi":"Urunum_yanlis_eksik_gönderildi/hata_bert_finetuned_model"
}

# Subreason modelleri dizini (alt modeller)
subreason_model_dirs = {
    "Ürünüm beklediğim kalitede değildi": "Urunum_bekledigim_kalitede_degil/alt_bert_finetuned_model",
    "Ürünün kutusu / ambalajı sorunluydu": "Urunun_kutusu_ambalaji_sorunluydu/alt_bert_finetuned_model",
    "Kargo firması / çalışanı / teslimat deneyiminden memnun değilim": "Kargo_firmasi_calisani_teslimat_deneyiminden_memnun_degilim/alt_bert_finetuned_model",
    "Mağaza çalışanı benimle ilgilenmedi / yeterli bilgiye sahip değildi": "Magaza_calisani_benimle_ilgilenmedi_yeterli_bilgiye_sahip_degildi/alt_bert_finetuned_model",
    "Kampanyalar ve kupon kodları konusunda sorun yaşadım": "Kampanyalar_ve_kupon_kodlari_konusunda_sorun_yasadim/alt_bert_finetuned_model",
    "Kasada uzun süre bekledim / tüm kasalar açık değildi": "Kasada_uzun_sure_bekledim_tum_kasalar_acik_degildi/alt_bert_finetuned_model",
    "Mağaza düzensiz / karışık / kirliydi": "Magaza_duzensiz_karisik_kirliydi/alt_bert_finetuned_model",
    "Ödeme işlemi sırasında sorun yaşadım": "Odeme_islemi_sirasinda_sorun_yasadim/alt_bert_finetuned_model",
    "Ürün / marka çeşidi azdı":"Urun_marka_çeşidi_azdı/alt_bert_finetuned_model",
    "Ürünüm geç teslim edildi": "Urunum_gec_teslim_edildi/alt_bert_finetuned_model",
    "Ürünüm yanlış / eksik gönderildi": "Urunum_yanlis_eksik_gönderildi/alt_bert_finetuned_model"
}

# Kuantizasyon ayarları
ov_config = OVConfig()
quant_config = PostTrainingQuantConfig(approach="static")


def quantize_and_save(model_dir: str):
    print(f"\nModel yükleniyor: {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    quantizer = OVQuantizer.from_pretrained(model, task="text-classification")

    # Her modelin kendi klasöründe openvino_model_quantized olarak kaydet
    save_directory = os.path.join(model_dir, "openvino_model_quantized")
    print(f"Kuantize model kaydedilecek: {save_directory}")

    quantizer.quantize(
        tokenizer=tokenizer,
        ov_config=ov_config,
        quantization_config=quant_config,
        calibration_dataset=None,  # Dilersen burada örnek bir dataset verebilirsin
        save_directory=save_directory,
        save_onnx_model=True
    )

    print(f"{model_dir} için kuantizasyon tamamlandı ve kaydedildi.")


if __name__ == "__main__":
    print("Reason modelleri için kuantizasyon başlıyor...")
    for reason, model_dir in reason_model_dirs.items():
        quantize_and_save(model_dir)

    print("\nSubreason modelleri için kuantizasyon başlıyor...")
    for reason, model_dir in subreason_model_dirs.items():
        quantize_and_save(model_dir)

    print("\nTüm modeller kuantize edildi ve kaydedildi.")
