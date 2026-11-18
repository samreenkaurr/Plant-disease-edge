# Results

All numbers come from a single training run per experiment (seed 0) on Google Colab (T4 GPU).
Items marked **TODO** have not been measured yet. Do not fill them in from memory.

## 1. Setup in one paragraph

Tomato and potato leaf images were drawn from two datasets that share six classes: PlantVillage (lab photos, plain
backgrounds) and PlantDoc (web-scraped field photos, cluttered backgrounds). PlantDoc has no healthy-potato class, so
`potato_healthy` was dropped and six classes remain: potato early blight, potato late blight, tomato early blight,
tomato late blight, tomato septoria leaf spot, tomato healthy. A MobileNetV3-Small pretrained on ImageNet was fine-tuned in two
phases (5 epochs with the backbone frozen, then up to 15 epochs with the top 40% unfrozen, early stopping on validation
accuracy, class-weighted loss, Adam at 1e-3 then 1e-4).

| Split | Source | Images |
|---|---|---|
| pv_train / pv_val | PlantVillage (90/10 split) | 7,442 / 829 |
| pd_train / pd_val | PlantDoc TRAIN (90/10 split) | 521 / 60 |
| pd_test | PlantDoc TEST (held out, **field benchmark**) | 54 |

| Experiment | Training data | Augmentation | Early-stopping set |
|---|---|---|---|
| E1 | PlantVillage | flip only | pv_val |
| E2 | PlantVillage | strong (crop/zoom, colour and lighting jitter, cutout) | pv_val |
| E3 | PlantVillage + PlantDoc | strong | pd_val (60 images) |

## 2. The lab-to-field gap

| Experiment | Lab accuracy (pv_val, n=829) | Field accuracy (pd_test, n=54) | 95% interval (Wilson) | Field macro-F1 |
|---|---|---|---|---|
| E1 | 98.6% | 40.7% (22/54) | 29 to 54% | 0.317 |
| E2 | 98.7% | 37.0% (20/54) | 25 to 50% | 0.316 |
| E3 | 99.5% | 53.7% (29/54) | 41 to 66% | 0.541 |

The lab-only baseline reaches about 99% on held-out lab images but drops by roughly 58 percentage points on field photos.
Strong augmentation alone did not help (E2 is 2 images below E1, well inside noise). Adding field photos to training (E3)
produced the highest field accuracy, 7 images more than E1.

**Caution.** The test set has 54 images, so the intervals are wide and E3's interval (41 to 66%) overlaps E1's (29 to 54%).
E3 scored higher in this run, but this experiment cannot show that the difference is statistically reliable. Only one seed
was run. The lab accuracy is also optimistic: PlantVillage contains near-duplicate photos of the same leaf, and the split
was random rather than grouped by leaf.

E3 fits its field training data (about 96% training accuracy) but stayed near 45% on the 60-image field validation set
during training, so it largely memorised the roughly 520 field images instead of generalising from them. Early stopping on
a 60-image set is itself noisy.

## 3. Calibration

| Experiment | ECE lab | ECE field (raw) | ECE field after temperature scaling | Fitted T |
|---|---|---|---|---|
| E1 | 0.016 | 0.478 | 0.154 | 9.14 |
| E2 | 0.008 | 0.487 | 0.089 | 10.00 |
| E3 | 0.008 | 0.187 | 0.183 | 2.02 |

The lab-only models are heavily overconfident on field images (ECE about 0.48). Temperature scaling (fitted on the 60-image
pd_val set) reduced this a lot for E1 and E2, but their fitted temperatures sit at or next to the search limit (10), so
those fits are unreliable. E3 was better calibrated to begin with and temperature scaling changed little (0.187 to 0.183).
The deployed temperature (2.02) was fitted on only 60 images, so treat the calibration results as indicative.

Selective prediction (answer only when confidence is above a threshold): **TODO** copy the coverage and accuracy at
thresholds 0.5 to 0.9 from `runs/e3_mixed/eval_pd_test/metrics.json` (`calibrated.selective`).

## 4. Compression and on-device model

Variants of the E3 model, evaluated on the same 54 field images.

| Variant | Size | Field accuracy | Macro-F1 | ECE |
|---|---|---|---|---|
| Keras (reference) | n/a | 53.7% | 0.541 | 0.187 |
| TFLite fp32 | 3.75 MB | 53.7% | 0.541 | 0.187 |
| TFLite fp16 | 1.93 MB | 53.7% | 0.541 | 0.212 |
| TFLite dynamic-range | 1.12 MB | 50.0% | 0.507 | 0.180 |
| TFLite int8 (post-training) | 1.23 MB | 35.2% | 0.336 | 0.271 |

fp16 is lossless on this test set at about half the size, and is the variant used in the app and demo. Dynamic-range
quantization lost 2 images, which is within noise. Full-integer post-training quantization lost 10 images (about 18 points),
and was only 0.1 MB smaller than the dynamic-range model. Its calibration set was 200 PlantDoc training images. A plausible
explanation is sensitivity of MobileNetV3's activations to integer quantization, but this was **not tested**. Quantization-aware
training or a larger, mixed-domain calibration set are untried alternatives.

## 5. What the models look at (Grad-CAM)

See `figures/gradcam_e1_vs_e3.png`: eight field images, four that E3 misclassified and four that it classified correctly.

- E3's attention was more sensible on some images where E1 failed. For a potato early-blight leaf, E1 attended to a corner of
  the image and called it tomato, while E3 attended to the lesion and was correct. For a septoria-spotted tomato leaf, E1
  attended to soil and was wrong at 0.94 confidence.
- E3 is not uniformly better. On a tomato early-blight image with straw mulch, E1 was correct (0.98) while E3 attended to
  straw and soil and was wrong. On a healthy tomato leaf held in a hand, E3's strongest attention was on dark foliage at the
  edge of the image.
- E1's errors carried confidence of 0.78 to 0.94, consistent with its poor calibration. Two of E3's correct predictions had
  low confidence (0.36 and 0.42) and would be flagged "not sure" by the app's 0.6 threshold.
- Most errors involve early versus late blight, or tomato versus potato.

These are eight selected images and Grad-CAM is a coarse attribution method, so this is illustrative and does not
establish how often either model uses the correct evidence. One image carries a stock-photo watermark, a reminder that
web-scraped data can contain shortcuts.

## 6. Demo and on-device performance

A web demo (`demo/server.py`) serves the fp16 model to a phone browser, tested on an iPhone over local Wi-Fi.
**The model runs on the laptop, not on the phone**, so it demonstrates the interface only.

On-device latency, memory and battery measurements on real phones: **TODO / not measured**. The Flutter app in `app/`
is written but has not been compiled or run, because the development Mac (Intel, macOS 14.4) could not run a current Xcode.

User study: **TODO / not done**.

## 7. Limitations

1. **Small field test set (54 images)** and a single seed per experiment. Differences of a few images are not reliable.
2. **Lab validation is optimistic** because of near-duplicate PlantVillage photos and a random, not leaf-grouped, split.
3. **Two crops, six classes only.** The model cannot recognise other plants or diseases and still outputs one of the six.
4. **Small, noisy field data** (about 520 training images, web-scraped, with possible label noise).
5. **Calibration fitted on 60 images**, and two of the fitted temperatures hit the search limit.
6. **No on-device measurements** and no user study (see section 6).
7. **int8 degradation unexplained**, and the proposed cause is untested.
8. Screening aid only, not a substitute for an agronomist. Field accuracy of about 54% is not good enough for decisions
   that matter.

## 8. Future work

Multiple seeds and a larger, leaf-grouped field test set. Oversampling the field images, or more field data, to reduce the
overfitting seen in E3. Retrying int8 with a mixed-domain calibration set or quantization-aware training. Compiling the
Flutter app and measuring latency on real devices. Out-of-distribution detection for plants outside the six classes.
