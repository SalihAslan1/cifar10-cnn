import argparse
import hashlib
import json
from html import escape
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
)

from data import DATASET_URL
from model import CLASSES, ROOT

INK = colors.HexColor("#162333")
MUTED = colors.HexColor("#485a6e")
BLUE = colors.HexColor("#1c4b73")
LINE = colors.HexColor("#d7e1ea")
PALE = colors.HexColor("#f1f6fa")
WIDTH = 170 * mm

font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(font_dir / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuBold", str(font_dir / "DejaVuSans-Bold.ttf")))

BODY = ParagraphStyle("Body", fontName="DejaVu", fontSize=9.5, leading=15, textColor=INK)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8, leading=11)
LABEL = ParagraphStyle("Label", parent=SMALL, fontName="DejaVuBold", textColor=BLUE)
TITLE = ParagraphStyle("Title", parent=BODY, fontName="DejaVuBold", fontSize=24, leading=30,
                       spaceAfter=12, textColor=BLUE)
HEADING = ParagraphStyle("Heading", parent=BODY, fontName="DejaVuBold", fontSize=15, leading=20,
                         spaceBefore=10, spaceAfter=8, textColor=BLUE)
CAPTION = ParagraphStyle("Caption", parent=SMALL, alignment=TA_CENTER, textColor=MUTED)


def para(value, style=BODY):
    return Paragraph(escape(str(value)), style)


def link(url):
    safe = escape(url, quote=True)
    return Paragraph(f'<link href="{safe}" color="#1c4b73">{safe}</link>', SMALL)


def table(rows, widths, header=False):
    styles = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
    ]
    if header:
        styles.append(("BACKGROUND", (0, 0), (-1, 0), PALE))
    return Table(rows, colWidths=widths, style=styles, hAlign="LEFT")


def illustration(path, width, height):
    original_width, original_height = ImageReader(str(path)).getSize()
    scale = min(width / original_width, height / original_height)
    return Image(str(path), width=original_width * scale, height=original_height * scale)


def percent(value):
    return f"%{value * 100:.2f}".replace(".", ",")


def count(value):
    return f"{value:,}".replace(",", ".")


def duration(seconds):
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes} dk {remainder:02d} sn"


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 10 * mm, "Bilgisayarlı Görü Dersi  /  CIFAR-10")
    canvas.drawRightString(190 * mm, 10 * mm, str(document.page))
    canvas.restoreState()


def phone_section(output, notes_path):
    section = [para("Telefon fotoğraflarıyla değerlendirme", HEADING)]
    phone_dir = output / "phone"
    manifest = phone_dir / "results.json"
    if not manifest.exists():
        return section + [
            para("Bu bölüm, öğrencinin kendi telefonuyla çektiği beş fotoğraf teslim "
                 "edildiğinde doldurulacak. Her fotoğraf için gerçek sınıf, model tahmini "
                 "ve tahminin skoru görüntünün yanında gösterilecek."),
        ]

    phone = json.loads(manifest.read_text(encoding="utf-8"))
    model_hash = hashlib.sha256((output / "best.pt").read_bytes()).hexdigest()
    if phone["checkpoint_sha256"] != model_hash:
        raise ValueError("Telefon sonuçları, bu rapordaki model dosyasıyla uyuşmuyor.")
    photos = phone["photos"]
    correct = sum(item["correct"] for item in photos)
    section += [para(f"Model beş telefon fotoğrafının {correct} tanesinde doğru sınıfı tahmin etti.")]
    rows = [[para(label, LABEL) for label in
             ("Fotoğraf", "Gerçek sınıf", "Tahmin ve skor", "Sonuç")]]
    for item in photos:
        photo = illustration(phone_dir / item["photo"], 40 * mm, 26 * mm)
        rows.append([
            [photo, para(item.get("source_name", item["photo"]), CAPTION)],
            para(item["ground_truth"], SMALL),
            para(f'{item["prediction"]} ({percent(item["probability"])})', SMALL),
            para("Doğru" if item["correct"] else "Hatalı", SMALL),
        ])
    section += [Spacer(1, 5 * mm), table(rows, [49 * mm, 38 * mm, 58 * mm, 25 * mm], header=True)]
    if notes_path:
        notes = notes_path.read_text(encoding="utf-8-sig").strip().split("\n\n")
        section += [para("Sonuçların yorumu", HEADING)]
        for note in notes:
            section += [para(note), Spacer(1, 2 * mm)]
    return section


def generate_report(output, destination, repository, phone_notes):
    training = json.loads((output / "training.json").read_text(encoding="utf-8"))
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    settings, environment = training["config"], training["environment"]
    inference = evaluation["inference"]
    scores = (
        f'Accuracy {percent(evaluation["accuracy"])} · Precision {percent(evaluation["macro_precision"])}'
        f' · Recall {percent(evaluation["macro_recall"])} · F1 {percent(evaluation["macro_f1"])}'
    )
    timing = (f'{inference["median_ms"]:.3f} ms medyan, '
              f'{inference["p95_ms"]:.3f} ms p95 (CPU)').replace(".", ",")
    title = "CIFAR-10 ile Görüntü Sınıflandırma"
    project_rows = [
        ("Project Title", title),
        ("Problem / Task", "Tek etiketli, 10 sınıflı görüntü sınıflandırma"),
        ("Dataset", "CIFAR-10 · 60.000 renkli görüntü"),
        ("Dataset URL", link(DATASET_URL)),
        ("Model Architecture", (
            "6 evrişim katmanlı CNN, 3 blok (64, 128, 256 kanal), "
            f'{count(training["parameters"])} eğitilebilir parametre'
        )),
        ("Input Size", "32 × 32 × 3 (RGB)"),
        ("Hyperparameters", (
            f'{settings["epochs"]} epoch, batch size {settings["batch_size"]}, '
            f'learning rate {settings["learning_rate"]:g} → {settings["min_learning_rate"]:g}, '
            f'weight decay {settings["weight_decay"]:g}, dropout 0,25, seed {settings["seed"]}'
        )),
        ("Optimizer", "AdamW ve kosinüs öğrenme oranı zamanlayıcısı"),
        ("Activation / Output Function", (
            "Gizli katmanlarda ReLU, çıkışta 10 logit. Tahminde softmax skorları kullanıldı."
        )),
        ("Loss Function", f'Cross entropy, label smoothing {settings["label_smoothing"]:g} (eğitim)'),
        ("Evaluation Metrics", "Accuracy, makro precision, makro recall ve makro F1"),
        ("Training Environment", (
            f'{environment["os"]}, Python {environment["python"]}, '
            f'PyTorch {environment["pytorch"]}, {environment["device"]}'
        )),
        ("Best Performance", (
            f'Doğrulama {percent(training["best_validation_accuracy"])} '
            f'(epoch {training["best_epoch"]}). Test kümesinde {scores}.'
        )),
        ("Inference Time", timing),
        ("Deployment Technology", "Gradio. Fotoğraf yükleme ve en yüksek üç sınıf skorunu gösterme."),
        ("Repository", link(repository) if repository else "Proje deposu bağlantısı teslimde eklenecek."),
    ]
    rows = [[para(label, LABEL), value if isinstance(value, Paragraph) else para(value, SMALL)]
            for label, value in project_rows]
    story = [
        para("BİLGİSAYARLI GÖRÜ ÖDEVİ", LABEL),
        Spacer(1, 4 * mm),
        para(title, TITLE),
        para("CIFAR-10 görüntülerini on sınıftan birine atayan bir evrişimli sinir ağı "
             "geliştirildi ve fotoğraf yüklemeyi destekleyen bir Gradio uygulamasına bağlandı."),
        Spacer(1, 5 * mm),
        table(rows, [53 * mm, 117 * mm]),
        PageBreak(),
        para("Veri hazırlığı ve eğitim", TITLE),
        para(f'CIFAR-10’un resmî eğitim kümesi, her sınıftan aynı sayıda örnek içerecek biçimde '
             f'{count(training["train_count"])} eğitim ve {count(training["validation_count"])} doğrulama '
             f'görüntüsüne ayrıldı. {count(evaluation["test_count"])} resmî test görüntüsü nihai '
             'değerlendirmede kullanıldı. RGB kanal ortalamaları ve standart sapmalar yalnızca '
             'eğitim bölümünden hesaplandı. Rastgele 32 × 32 kırpma (4 piksel dolgu) ve %50 '
             'olasılıkla yatay çevirme eğitim görüntülerine uygulandı.'),
        para("Mimari ve eğitim kararları", HEADING),
        para("Her blok iki adet 3 × 3 evrişim, batch normalization, ReLU ve bir max pooling "
             "katmanı içerir. Kanal sayısı derinlikle artar. Global ortalama havuzlama öznitelik "
             "haritalarını özetler, dropout düzenlileştirme sağlar ve doğrusal katman on logit "
             "üretir. AdamW uyarlamalı adım ve ayrık ağırlık azalması sağlar. Kosinüs zamanlayıcı "
             "eğitimin sonuna doğru adımı küçültür. Cross entropy çok sınıflı tahmin için "
             "kullanıldı. Label smoothing eğitim hedeflerini yumuşatır."),
        Spacer(1, 4 * mm),
        illustration(output / "training_curves.png", WIDTH, 71 * mm),
        para("Epoch başına eğitim ve doğrulama kaybı ile doğruluğu", CAPTION),
        para("Eğitim süresi ve model seçimi", HEADING),
        para(f'Toplam {duration(training["training_seconds"])}. Ölçüm epoch içindeki eğitim ve '
             'doğrulamayı kapsar. Veri indirme, veri hazırlama ve dosyaya checkpoint yazma hariçtir. '
             f'{environment["device"]} üzerinde {environment["precision"]} kullanıldı. '
             f'En yüksek doğrulama doğruluğu {training["best_epoch"]}. epoch’ta '
             f'{percent(training["best_validation_accuracy"])} olarak ölçüldü. '
             'Bu epoch’un ağırlıkları test ve uygulama için kaydedildi.'),
        PageBreak(),
        para("Test kümesi sonuçları", TITLE),
        para(f'{count(evaluation["test_count"])} görüntüde {scores}. Makro ortalamada her '
             f'sınıfa eşit ağırlık verildi. Test çalıştırması {evaluation["evaluation_device"]} '
             f'üzerinde {evaluation["evaluation_precision"]} ile yapıldı.'),
        Spacer(1, 5 * mm),
    ]
    class_rows = [[para(label, LABEL) for label in ("Sınıf", "Precision", "Recall", "F1")]]
    class_rows.extend([[para(row["class"], SMALL)] + [
        para(percent(row[key]), SMALL) for key in ("precision", "recall", "f1")
    ] for row in evaluation["per_class"]])
    class_table = table(class_rows, [23 * mm, 21 * mm, 18 * mm, 18 * mm], header=True)
    confusion = illustration(output / "confusion_matrix.png", 88 * mm, 84 * mm)
    story.append(Table([[class_table, confusion]], colWidths=[82 * mm, 88 * mm],
                       style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    matrix = evaluation["confusion_matrix"]
    occurrences, actual, predicted = max(
        (n, i, j) for i, row in enumerate(matrix) for j, n in enumerate(row) if i != j
    )
    story += [
        Spacer(1, 5 * mm),
        para(f'En sık yönlü hata, {occurrences} örnekte “{CLASSES[actual]}” sınıfının '
             f'“{CLASSES[predicted]}” olarak tahmin edilmesiydi.'),
        para("Tahmin süresi ve uygulama", HEADING),
        para(f'{timing}. Ölçüm, {inference["warmup_runs"]} ısınma tahmininden sonra '
             f'{inference["samples"]} CIFAR-10 test görüntüsünde tek tek yapıldı. '
             f'{inference["cpu"]}, {inference["threads"]} iş parçacığı kullandı. '
             'PIL görüntüsünden sınıf skorlarına '
             'kadar geçen süre ölçüldü. Dosya okuma ve tarayıcı gecikmesi ölçüme dahil edilmedi. '
             'Uygulama, telefon görsellerinin EXIF yönünü düzeltir, görüntüyü RGB’ye çevirir '
             've merkezden kare kırparak 32 × 32 boyutuna getirir.'),
    ]
    if (output / "phone" / "results.json").exists():
        story.append(PageBreak())
    story.extend(phone_section(output, phone_notes))
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(destination), pagesize=A4, title=title,
                                 leftMargin=20 * mm, rightMargin=20 * mm,
                                 topMargin=17 * mm, bottomMargin=21 * mm)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Rapor yazıldı: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ödevin PDF raporunu üret")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--pdf", type=Path, default=ROOT / "CIFAR-10 Proje Raporu.pdf")
    parser.add_argument("--repository")
    parser.add_argument("--phone-notes", type=Path)
    args = parser.parse_args()
    generate_report(args.output, args.pdf, args.repository, args.phone_notes)
