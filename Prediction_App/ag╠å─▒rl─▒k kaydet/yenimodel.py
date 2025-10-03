from optimum.intel.openvino import OVModelForSequenceClassification
from transformers import AutoTokenizer, pipeline


def test_openvino_model():
    # Model ve tokenizer yolları
    model_dir = "Urunum_bekledigim_kalitede_degil/hata/openvino_model_quantized"
    tokenizer_dir = "Urunum_bekledigim_kalitede_degil/hata_bert_finetuned_model"

    # Tokenizer ve model yükleniyor
    print("Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    print("OpenVINO modeli yükleniyor...")
    model = OVModelForSequenceClassification.from_pretrained(model_dir)

    # Inference pipeline kuruluyor
    print("Pipeline kuruluyor...")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Test verisi
    test_text = "Kargo firması."

    # Tahmin
    print("Model tahmin yapıyor...")
    result = classifier(test_text)
    print("Ham Sonuç:", result)

    # Label-to-subreason eşlemesi
    label_map = {
        0: "Doğru",
        1: "Yanlış",

    }

    # Etiket çözümlemesi
    label_id = int(result[0]["label"].split("_")[1])
    print("Alt Reason Tahmini:", label_map[label_id])


if __name__ == "__main__":
    test_openvino_model()
