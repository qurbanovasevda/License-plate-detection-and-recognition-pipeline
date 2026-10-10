# Nömrə lövhəsinin aşkarlanması və oxunması (License Plate Detection & Recognition Pipeline)

Bu layihə iki ayrı modeli bir pipeline-da birləşdirir: **YOLOv8** şəkildə nömrə lövhəsini tapır, **EasyOCR** kəsilmiş lövhədən mətni oxuyur, qayda əsaslı təmizləmə isə nəticəni Azərbaycan nömrə formatına (`10AA123`) uyğunlaşdırır. Hər iki mərhələ əl ilə etiketlənmiş test setində ayrıca qiymətləndirilir (detektor: IoU; OCR: exact match və CER).

## 1. Pipeline necə işləyir

```
şəkil
  → YOLOv8 aşkarlama (imgsz=640; heç nə tapılmasa imgsz=1280 ilə təkrar)
  → daxili-box təmizləmə (başqa box-un içində qalan fraqmentlər atılır)
  → 10% padding ilə crop (şəkil sərhədindən çıxmır)
  → preprocess (boz + 2x böyütmə + CLAHE)
  → EasyOCR (yalnız A-Z və 0-9; seqmentlər soldan sağa düzülür)
  → mətn təmizləmə (artıq simvolları at, mövqeyə görə 0/O, 1/I və s. düzəliş)
  → lövhə mətni
```

İstifadə:

```python
from pipeline import read_plate, read_plate_text

read_plate("data/unseen/unseen3.jpg")
# [{'box': (698, 557, 931, 618), 'conf': 0.685, 'raw': '39BC205', 'text': '39BC205'}]

read_plate_text("data/unseen/unseen3.jpg")   # '39BC205'
```

- Lövhə tapılmasa xəta atılmır, boş siyahı (`[]`) və ya boş mətn qaytarılır.
- Bir şəkildə bir neçə lövhə olarsa, hər biri ayrıca kəsilir və oxunur.
- Qaytarılan `box` detektorun öz orijinal box-udur (padding yalnız OCR üçün crop-a aiddir). Box-lar heç vaxt əllə çəkilmir, hamısı modelin çıxışıdır.

## 2. Texniki seçimlər

**Detektor: YOLOv8 (Ultralytics).** Fine-tune üçün ən rahat və tapşırıqda tövsiyə olunan variantdır. Haar cascade istifadə olunmayıb.
Model Google Colab-da Roboflow Universe-dəki ictimai *license plate detection* datasetində öyrədilib (YOLO formatı; müxtəlif ölkələrin nömrələri qarışıqdır). Nəticə `weights/best.pt` faylıdır.

| Parametr | Dəyər |
|---|---|
| Model | YOLOv8n (`yolov8n.pt`-dən fine-tune; sürətli, yüngül, CPU-da aşkarlama üçün kifayət qədər) |
| Dataset | [license-plate-detection (Haeun Kim, Roboflow Universe, v2)](https://universe.roboflow.com/haeun-kim-ri91b/license-plate-detection-wienp/dataset/2), lisenziya CC BY 4.0 |
| Train şəkilləri | 5115 (valid: 225) |
| Epoch / imgsz | 20 / 640 |
| Valid nəticəsi (`best.pt`) | mAP50 0.96, precision 0.968, recall 0.927, mAP50-95 0.668 |
| Label hazırlığı | Datasetdə box və poliqon formatı qarışıq idi (train-də 2921 poliqon). Poliqonlar training-dən əvvəl onları əhatə edən box-a çevrildi. |

Valid nəticəsi yalnız `best.pt`-nin seçilməsi üçündür; modelin əsl qiymətləndirməsi aşağıdakı ayrıca test setindədir (bölmə 5).
Training qrafikləri: `reports/training/` (confusion matrix, results, val_batch0_pred).

**OCR: EasyOCR.** Tesseract sistem səviyyəsində quraşdırma və Windows-da PATH/konfiqurasiya tələb edir. EasyOCR isə yalnız `pip` ilə qurulur (daha ağır yükləmə: ilk işə salanda modeli internetdən endirir). Bu layihədə CPU üzərində işləyir.

## 3. Datalar

| | Mənbə | Say |
|---|---|---|
| Train | Roboflow Universe ictimai dataset (qarışıq ölkələr) | 5115 |
| Test (held-out) | Azərbaycan nömrəli şəkillər: şəkillər həm internetdən götürülüb həm də öz çəkdiyimdir  | 25 şəkil, 30 lövhə box-u |
| Unseen (anti-cheat) | Test və train-də olmayan 3 yeni şəkil | 3 |

- Test şəkilləri train datasından **tam ayrıdır**. İlk versiyada test və train eyni data idi, bu düzəldildi.
- Box-lar **əllə** Roboflow-da çəkilib (YOLO formatı): `data/test/images/`, `data/test/labels/`.
- Mətn etiketləri `data/test/plates.csv` faylındadır: hər şəkil üçün **əsas (ən böyük) lövhənin** mətni. `images19` oxunmadığı üçün `UNREADABLE` yazılıb və OCR hesabından çıxarılıb (OCR üçün 24 şəkil). `images23` və `images24` eyni maşının ön və arxa şəkilləridir.
- Çox lövhəli şəkillərdə bütün lövhələr detection (IoU) hesabına daxildir, OCR yalnız əsas lövhə üzərində ölçülür.

## 4. Metrikalar

- **IoU:** GT box ilə modelin box-u arasında. Eşik 0.5. Hər GT box ən yaxşı IoU-lu proqnozla bir-bir uyğunlaşdırılır.
- **Recall@0.5 / Precision@0.5:** tapılan GT box-ların və düzgün proqnozların payı.
- **Mean IoU:** (a) bütün GT box-lar üzrə (qaçırılanlar 0 sayılır), (b) yalnız tapılanlar üzrə.
- **Exact match:** təmizlənmiş OCR mətni real mətnlə hərfi-hərfinə eynidir.
- **CER:** Levenshtein məsafəsi / real mətnin uzunluğu. Detektorun qaçırdığı lövhə end-to-end hesabda CER = 1 sayılır.

## 5. Nəticələr (test seti, 25 şəkil)

**Detektor**

| Metrika | Dəyər |
|---|---|
| Recall@0.5 | 0.800 (24/30 box) |
| Precision@0.5 | 0.960 |
| Mean IoU (bütün GT, qaçırılanlar daxil) | 0.606 |
| Mean IoU (tapılanlar) | 0.757 |

**OCR (EasyOCR + təmizləmə)**

| Metrika | Dəyər |
|---|---|
| End-to-end exact match | 11/24 = 0.458 |
| End-to-end mean CER | 0.345 |
| Detektorun tapdığı lövhələr üzrə exact match | 11/22 = 0.500 |
| Detektorun tapdığı lövhələr üzrə mean CER | 0.286 |

**Konfiqurasiya müqayisəsi (hamısı test setində)**

| Konfiqurasiya | Recall | Precision | Mean IoU (bütün GT) | OCR exact | Mean CER |
|---|---|---|---|---|---|
| İlk versiya (padding-li box ilə IoU, sadə AZ-zolaq qaydası) | 0.633 | 0.792 | 0.483 | 6/24 | 0.512 |
| imgsz 640, conf 0.25 | 0.733 | 0.917 | 0.556 | 9/24 | 0.429 |
| imgsz 640, conf 0.10 | 0.767 | 0.793 | 0.591 | 10/24 | 0.387 |
| imgsz 1280, conf 0.25 | 0.667 | 0.690 | 0.512 | 9/24 | 0.458 |
| **640 → 1280 ehtiyat, conf 0.25** | **0.800** | **0.889** | **0.606** | **11/24** | **0.345** |
| yekun: yuxarıdakı + daxili-box təmizləmə | 0.800 | **0.960** | 0.606 | 11/24 | 0.345 |

Müşahidələr:
- İlk versiyada IoU padding-li box ilə hesablanırdı və mükəmməl aşkarlamada belə təxminən 0.69-dan yuxarı çıxa bilmirdi. Düzəlişdən sonra IoU detektorun öz box-u ilə ölçülür.
- `conf=0.10` bir lövhə əlavə tapdı, amma yalançı box-lar çoxaldı (precision 0.917 → 0.793). `conf=0.25` saxlanıldı.
- `imgsz=1280` kiçik şəkillərdə (məs. 300×300) lövhələri tapır, amma böyük şəkillərdə əvvəl tapılanları itirir və çox yalançı box gətirir. Ona görə 1280 yalnız 640-da heç nə tapılmayanda ehtiyat kimi istifadə olunur.

**OCR preprocess müqayisəsi** (detektorun tapdığı 20 lövhə, o zamankı vəziyyətdə):

| Preprocess | Exact match | Mean CER |
|---|---|---|
| **Boz + 2x + CLAHE (seçilən)** | **9/20** | **0.314** |
| Boz 3x (CLAHE-siz) | 8/20 | 0.386 |
| Hündürlük 100px + kənar boşluğu | 7/20 | 0.436 |
| Xam rəngli crop | 5/20 | 0.464 |

## 6. Mətn təmizləmə qaydaları (`pipeline.py`)

1. Hərf və rəqəmdən başqa hər şey silinir, hərflər böyüdülür (`99-TH 147` → `99TH147`).
2. EasyOCR yalnız `A-Z0-9` simvollarına icazə verir (`allowlist`), seqmentlər x koordinatına görə soldan sağa düzülür (çəp lövhələrdə sıra qarışa bilir).
3. 7 simvoldan uzundursa (məs. mavi `AZ` zolağı oxunubsa), formata ən yaxın 7 simvol seçilir. Əvvəl baş/son simvol atmağa, yalnız format tam uyğun gəlmirsə ortadan atmağa baxılır.
4. Mövqeyə görə düzəliş (Azərbaycan formatı: 2 rəqəm + 2 hərf + 3 rəqəm). Rəqəm yerində `O→0, Q→0, I→1, L→1, S→5, B→8, Z→2`, hərf yerində `0→O, 1→I, 5→S, 8→B, 2→Z, 4→A`.

## 7. Crop yoxlaması (çox maşınlı şəkillər)

Aşağıdakı üç şəkildə kadrda bir neçə maşın var. Crop-lar yalnız detektorun tapdığı box-lardan, 10% padding ilə və şəkil sərhədini aşmadan kəsilir:

| Şəkil | Kadrda | Etiketli lövhə / tapılan | Crop faylları |
|---|---|---|---|
| `images5` | 3 maşın | 3 / 1 (kiçik lövhələr qaçırıldı) | `reports/crops/images5_*.jpg` |
| `images12` | 2 maşın | 2 / 2 | `reports/crops/images12_*.jpg` |
| `images7` | əsas maşın + arxa fonda bir neçə maşın | 1 / 1 (arxa fondakı lövhələr çox kiçikdir, etiketlənməyib) | `reports/crops/images7_*.jpg` |

Bütün şəkillər üzərində box-ların görüntüsü: `reports/vis/` (yaşıl: əl ilə etiket, qırmızı: model).
`images11` sıfır aşkarlama halına nümunədir (pipeline xəta vermir, boş nəticə qaytarır). `images4` isə bir lövhənin iki fraqmentə bölündüyü hala nümunədir (bax: uğursuzluq təhlili).

## 8. End-to-end dəqiqlik cədvəli

`image → aşkarlanmış box → OCR mətni → düzgün/səhv` (test seti, `python make_report.py` ilə avtomatik doldurulur):

<!-- TABLE_START -->
| Şəkil | GT box | Aşkarlanmış box | OCR (xam) | OCR (təmiz) | Real mətn | IoU | Nəticə |
|---|---|---|---|---|---|---|---|
| `images.jpg` | 1 | (148, 189, 223, 209) | 99SM005 | 99SM005 | 99SM005 | 0.692 | ✅ düzgün |
| `images10.jpg` | 1 | (46, 38, 166, 123) | 62BE 0381 | 62BE038 | 62BE038 | 0.819 | ✅ düzgün |
| `images11.jpg` | 2 | — | — | — | 88DJ270 | 0.0 | ❌ detektor qaçırdı |
| `images12.jpg` | 2 | (127, 319, 196, 342) | — | — | 90YL736 | 0.747 | ❌ OCR səhvi |
| `images13.jpg` | 1 | (99, 245, 169, 272) | DL9 | DL9 | 10LM999 | 0.654 | ❌ OCR səhvi |
| `images14.jpg` | 1 | (68, 65, 137, 88) | 720GH846 | 20GH846 | 20GH846 | 0.884 | ✅ düzgün |
| `images15.jpg` | 1 | (89, 25, 147, 42) | TS | TS | 99ES190 | 0.862 | ❌ OCR səhvi |
| `images16.jpg` | 1 | (143, 70, 203, 93) | ICS | ICS | 77CD055 | 0.757 | ❌ OCR səhvi |
| `images17.jpg` | 1 | (60, 47, 145, 62) | 99AA522 | 99AA522 | 99AA522 | 0.663 | ✅ düzgün |
| `images18.jpg` | 1 | (72, 47, 161, 66) | 399AB017 | 99AB017 | 99AB017 | 0.806 | ✅ düzgün |
| `images19.jpg` | 1 | — | — | — | UNREADABLE | 0.0 | — (oxunmur, hesaba daxil deyil) |
| `images2.jpg` | 1 | (192, 303, 410, 360) | 100010 | 100010 | 10QO100 | 0.674 | ❌ OCR səhvi |
| `images20.jpg` | 1 | (95, 210, 154, 228) | E974388 | E97A388 | 90ZH388 | 0.781 | ❌ OCR səhvi |
| `images21.jpg` | 1 | — | — | — | 99UD290 | 0.0 | ❌ detektor qaçırdı |
| `images22.jpg` | 1 | (246, 310, 319, 354) | 9904522 | 99OA522 | 99OA522 | 0.796 | ✅ düzgün |
| `images23.jpg` | 1 | (102, 394, 250, 431) | 50DC3671 | 50DC367 | 50DC367 | 0.778 | ✅ düzgün |
| `images24.jpg` | 1 | (209, 166, 332, 203) | 50DC3671 | 50DC367 | 50DC367 | 0.771 | ✅ düzgün |
| `images25.jpg` | 1 | (190, 212, 346, 252) | 90RB214 | 90RB214 | 90RB214 | 0.802 | ✅ düzgün |
| `images3.jpg` | 1 | (187, 129, 349, 181) | 9OLA467 | 90LA467 | 90LA467 | 0.84 | ✅ düzgün |
| `images4.jpg` | 2 | (314, 334, 406, 357) | 90XHL7L | 90XH171 | 90XH474 | 0.681 | ❌ OCR səhvi |
| `images5.jpg` | 3 | (144, 192, 241, 211) | 994THIZZ | 94TH122 | 99TH147 | 0.833 | ❌ OCR səhvi |
| `images6.jpg` | 1 | (326, 356, 421, 398) | 90 SD022 | 90SD022 | 90SD024 | 0.742 | ❌ OCR səhvi |
| `images7.jpg` | 1 | (837, 811, 963, 863) | E7D | E7D | 77PF906 | 0.61 | ❌ OCR səhvi |
| `images8.jpg` | 1 | (224, 681, 427, 761) | 99GO196 | 99GO196 | 99GO196 | 0.832 | ✅ düzgün |
| `images9.jgp.jpg` | 1 | (18, 568, 205, 660) | 90HM 24 | 90HM24 | 90HM224 | 0.822 | ❌ OCR səhvi |
<!-- TABLE_END -->

**Ümumi dəqiqlik (OCR, 24 oxunan şəkil üzrə):** 11/24 lövhə tam düzgün oxunub (0.458). Detektor 24 lövhədən 22-sini tapıb, tapılanların 11-i düzgün oxunub (0.500).

## 9. Uğursuzluq təhlili

| Şəkil | Mərhələ | Nə baş verdi | Ehtimal olunan səbəb |
|---|---|---|---|
| `images21` | Detektor | Heç bir lövhə tapılmadı | Lövhə bamperdə aşağıda, güclü bucaqla və kölgədədir. imgsz=1280-də ən yüksək box lövhəyə deyil, farın yerinə düşür (yalançı pozitiv). |
| `images11` | Detektor | Heç bir lövhə tapılmadı | Əsas lövhə kadrın kənarında və güclü bucaqlıdır, arxa fondakı ikinci lövhə çox kiçikdir. |
| `images4` | Detektor + OCR | Lövhə iki fraqment box-a bölündü (`90X`, `H474`), OCR `90XH171` oxudu (`90XH474` olmalı idi) | Detektor lövhəni bütöv deyil, iki hissə kimi tapdı; hər crop-da mətn yarımçıqdır. |
| `images5` | OCR | `99TH147` → `94TH122` | Lövhə aydındır və düzgün tapılıb (conf 0.52), amma crop cəmi təxminən 97×19 px-dir; `4` və `7` rəqəmləri `Z`/`2` kimi oxunub. |
| `images12`, `images15`, `images16`, `images13` | OCR | Nəticə boş, `TS`, `ICS`, `DL9` | Çox kiçik crop-lar (17-27 px hündürlük), OCR-in oxuya biləcəyi detal yoxdur. |
| `images7` | OCR | `77PF906` → `E7D` | Axşam işığında az işıqlı, kiçik lövhə; box düzgün tapılıb (IoU 0.61, conf 0.46), amma crop-da simvollar seçilmir. |
| `images20` | OCR | `90ZH388` → `E97A388` | Lövhənin sol hissəsi (`90ZH`) səhv oxunub, `4→A` qaydası da əlavə səhvə səbəb olub. |
| `images9` | OCR | `90HM224` → `90HM24` | Yan-yana eyni `22` rəqəmi bir simvol kimi oxunub (CTC dekoderinin təkrar simvolu birləşdirməsi). |
| `images2` | OCR | `10QO100` → `100010` | `Q`, `O`, `0` qarışıb; nəticə 6 simvol olduğu üçün mövqe düzəlişi tətbiq olunmur. |
| `images6`, `images4`, `images5` | OCR | `024`, `474`, `147` → `022`, `171`, `122` | Hər üçündə rəqəm `4` səhv oxunub (`2`, `L`, `Z`). EasyOCR-ın ümumi modeli bu lövhə fontunda `4`-ü zəif tanıyır. |

Ümumi nəticə: OCR xətalarının çoxu kiçik crop-lardan və EasyOCR-ın ümumi ingilis modelinin lövhə fontuna (xüsusən `4`) uyğunlaşmamasından qaynaqlanır. Detektor xətaları isə güclü bucaqlı, kölgədə və ya kadr kənarında qalan lövhələrdə cəmləşir.

## 10. Görünməmiş (unseen) şəkillərdə yoxlama

Real nömrələr özüm tərəfindən oxunub. Bu şəkillər train və test datasında yoxdur. Skript: `python run_unseen.py`.

| Şəkil | Real | Düzəlişlərdən əvvəl | Düzəlişlərdən sonra |
|---|---|---|---|
| `unseen1` | `10ZM477` | `02MZ771` ❌ | `10ZM771` ❌ (`4` rəqəmi `Z` kimi oxunub) |
| `unseen2` | `77DV120` | `12O77DV` ❌ (seqmentlərin sırası qarışıb) | `77DV120` ✅ |
| `unseen3` | `39BC205` | `39BC205` ✅ (amma 3 box) | `39BC205` ✅ (1 box) |

Detektor 3 şəkildə 3 lövhəni də tapdı. OCR əvvəl 1/3, düzəlişlərdən sonra 2/3 düzgün oxudu.

Qeyd: bu şəkillər düzəlişləri (daxili-box təmizləmə, x-ə görə sıralama, `strip_country_tag`-ın yenilənməsi) tapmaq üçün də istifadə olunub, ona görə düzəlişdən sonrakı nəticə tam "təmiz" yoxlama sayılmamalıdır. Düzəlişlərin hansının nə dərəcədə təsdiqləndiyi 11-ci bölmədə yazılıb. Ümumi yoxlama üçün düzəlişlərə təsir etməyən yeni şəkillərdə ayrıca sınaq planlaşdırılır.

## 11. Dürüstlük qeydi

Aşağıdakı qərarlar test və ya unseen nəticələrinə baxandan sonra verilib, yəni qismən həmin şəkillər üzərində tənzimlənib. Hər birinin nə dərəcədə təsdiqləndiyi də göstərilib:

| Dəyişiklik | Nəyə baxaraq edilib | Sübut |
|---|---|---|
| IoU-nun detektorun orijinal box-u ilə hesablanması | Test nəticələri | Metodoloji düzəliş (padding-li box IoU-nu süni aşağı salırdı) |
| 640 → 1280 ehtiyat strategiyası, `conf=0.25` seçimi | Test nəticələri | Test setində ölçülüb (cədvəl bölmə 5) |
| Preprocess seçimi | Test nəticələri | Test setində 4 variant müqayisə olunub (bölmə 5) |
| `4→A` xəritəsi | `images22` | Yalnız bir şəkil, ümumi sübut zəifdir |
| Daxili-box təmizləmə (`suppress_nested`) | `unseen3` | Test setində də ölçüldü: precision 0.889 → 0.960, recall dəyişmədi. Unseen-dən asılı deyil |
| OCR seqmentlərinin x-ə görə sıralanması | `unseen2` | Prinsip olaraq məntiqlidir (çəp lövhədə EasyOCR sırası qarışır), test setində nəticə dəyişmədi, yeganə təsdiq bir nümunədir. İki sətirli lövhələrdə səhv işləyər |
| `strip_country_tag`-ın "ortadan simvol atma" hissəsi | `unseen1` | Ölçülə bilən fayda yoxdur: `unseen1` yenə səhv qaldı, test nəticələri dəyişmədi |

Test seti kiçikdir (25 şəkil), ona görə rəqəmlər statistik cəhətdən güclü sübut deyil, istiqamət göstəricisidir. Düzəlişlərdən birinin (dəyişdirilmiş simvol seçmə qaydası) OCR dəqiqliyini müvəqqəti 11/24-dən 8/24-ə salması test seti ilə tutulub və düzəldilib.

## 12. Məhdudiyyətlər və gələcək iş

- Train datasetində müxtəlif ölkələrin nömrələri var, Azərbaycan lövhələri az təmsil olunub. Bucaqlı, kölgədə və kadr kənarında qalan lövhələrdə detektor zəifdir. Test şəkillərindən kənar Azərbaycan şəkilləri ilə əlavə fine-tune planlaşdırıla bilər.
- Təmizləmə yalnız 7 simvollu `10AA123` formatına uyğunlaşdırılıb (digər formatlar düzəliş almır).
- EasyOCR ümumi məqsədli modeldir; lövhə fontuna xüsusi öyrədilmiş OCR və ya çox kiçik crop-lar üçün super-resolution nəticəni yaxşılaşdıra bilər.
- Bonus: Tesseract ilə eyni test setində müqayisə hələ aparılmayıb.

## 13. Layihə strukturu

```
plate-recognition-pipeline/
├── pipeline.py          # detektor + crop + OCR + təmizləmə
├── evaluate.py          # IoU, exact match, CER, nəticə cədvəli
├── visualize.py         # GT və proqnoz box-larını çəkir, crop-ları saxlayır
├── run_unseen.py        # data/unseen şəkillərində pipeline
├── make_report.py       # README-dəki dəqiqlik cədvəlini avtomatik doldurur
├── check_data.py        # şəkil/label adlarının uyğunluğunu yoxlayır
├── rename_labels.py     # Roboflow label adlarını şəkil adlarına uyğunlaşdırır
├── ocr_ablation.py      # preprocess variantlarının müqayisəsi
├── debug_plate.py       # tək şəkil üçün diaqnostika
├── requirements.txt
├── .gitignore
├── weights/best.pt
├── data/test/{images,labels,plates.csv}
├── data/unseen/
└── reports/             # nəticə CSV-ləri, training qrafikləri, vis/, crops/, unseen/
```

## 14. Quraşdırma və işlətmə

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

```
python pipeline.py data/unseen/unseen3.jpg   # tək şəkil
python evaluate.py                           # test setində qiymətləndirmə
python visualize.py                          # box və crop şəkilləri
python run_unseen.py                         # data/unseen qovluğu
python make_report.py                        # README cədvəlini yenilə
```

Python 3.12 ilə sınanıb. Bütün əmrlər layihənin kök qovluğundan işlədilməlidir (yollar nisbidir). İlk işə salanda EasyOCR modellərini internetdən endirir.
