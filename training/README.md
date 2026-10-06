# Training a more accurate face-expression model

`face_expression_kfold.ipynb` trains the model that the Django app uses. It replaces the previous
YOLO11-cls / 48×48 models with a pipeline built around the real accuracy bottlenecks:

| Step | What it does | Why it matters |
|---|---|---|
| Face crops | Crops every photo with YuNet (the detector the app already uses). Images smaller than 100 px, like the 48×48 FER faces, already are face crops. | At 224 px, a face in a full photo is only a few pixels wide. |
| Pretrained on faces | **HSEmotion** EfficientNet is trained on AffectNet (450k faces). Its output layer starts from the matching AffectNet emotions, so it works before any training. If it can't be loaded, the notebook falls back to ConvNeXt-V2 automatically. | ImageNet backbones have never learned expressions. |
| Clean evaluation | A fixed test set plus 5-fold cross-validation. Splits are by "picture group" (md5 duplicates and perceptual-hash near-duplicates stay together). | A single split of a few hundred faces is noisy by ±6 points, and duplicates leak between train and test. |
| Regularisation | Training uses only the originals (offline `__aug` copies are optional), plus online augmentation, mixup, label smoothing, EMA weights and early stopping. | Training on 15–25 near-identical copies of each face teaches the model to memorise. |
| Ensemble | The 5 fold models, and optionally several backbones, are averaged. | Usually adds 1–3 points of accuracy. |
| Label cleaning | An out-of-fold mislabel finder produces `mislabel_candidates.csv` and a contact sheet; your corrections go in `label_fixes.csv`. | Expression datasets typically have 5–10% wrong labels. |
| Grayscale | Detected automatically. Most of `Data/` is gray, so every image is converted to gray, including inside the ONNX model. | Stops colour from becoming a shortcut, since only a few photos are in colour. |
| Deployment | Exports one `fer_ensemble.onnx` (normalisation, grayscale, softmax and flip-TTA are built in) plus `deployment_metadata.json`, and installs them in `face_expression_model/`. | The app picks the model up automatically, with no code changes. |

## Quick start (Windows, NVIDIA GPU)

```bat
cd path\to\face_expression
python -m venv .venv && .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r training\requirements-train.txt
cd training
jupyter notebook face_expression_kfold.ipynb
```

Run all cells. With the default settings (`Data/`, `hsemotion_b0`) the notebook:

1. Indexes `Data/<class>/`. Class names come from the folder names.
2. Builds face crops into `work/fer_crops/` (cached).
3. Trains 5 folds, saving to `artifacts/fer_kfold/hsemotion_b0/fold*/best.pt`. A fold that has already finished is skipped when you re-run.
4. Reports out-of-fold and test accuracy (with a 95% confidence interval), and saves the confusion matrix and mislabel candidates.
5. Exports `artifacts/fer_kfold/export/fer_ensemble.onnx` and copies it into `face_expression_model/`.

Then restart the app (`python manage.py runserver`). The new model becomes the default in the
dropdown, and the old models stay selectable.

**First run:** HSEmotion weights (~16 MB for B0, ~30 MB for B2) download from GitHub into
`work/weights/`. If your network blocks GitHub, download `enet_b0_8_best_afew.pt` /
`enet_b2_8.pt` by hand from
<https://github.com/HSE-asavchenko/face-emotion-recognition/tree/main/models/affectnet_emotions>
and put them in `work/weights/`. timm backbones download from Hugging Face.

**Time on an RTX 5070 Ti, full `Data/` (~59k images):** cropping takes a few minutes once.
`hsemotion_b0` takes roughly 2–4 min per epoch, and `EPOCHS="auto"` uses 12 epochs for large
datasets. To get a quick first number, set `FOLDS_TO_RUN = [0]`.

## Improving accuracy step by step

1. **Fix labels.** Open `artifacts/fer_kfold/<backbone>/mislabel_candidates.jpg`. In
   `mislabel_candidates.csv`, set `action` to `drop` or `relabel` (and fill in `new_label`) and save
   it as `label_fixes.csv` in the repo root. Re-run. Your image folders are never modified. Also
   check `no_face_detected.csv`: photos with no face, like the silhouette pictures in `Data/Happy`,
   should be dropped.
2. **Train a second backbone.** Set `BACKBONE = "hsemotion_b2"` (or `"convnextv2_tiny"`) and run
   again from section 5. Section 9 picks the best combination using out-of-fold accuracy, and
   section 10 exports it.
3. **Tune.** Use `TRAIN_OVERRIDES`, e.g. `{"lr_backbone": 5e-5}` or `{"mixup_alpha": 0.0}`. Compare
   results on out-of-fold accuracy, never on the test set.

## Using the 6-class `augmented_image` dataset instead

Set `SOURCE_DIR = Path(r"D:\Shitol\augmented_image")`. Files named `*__aug<k>.*` are recognised as
offline copies of their original. They are ignored by default, or used for training only with
`OFFLINE_COPIES_PER_ORIGINAL = k`. Note that the web UI's radar chart currently lists the 5
classes of `Data/`.

## Files

| File | Purpose |
|---|---|
| `face_expression_kfold.ipynb` | The notebook: data → crops → K-fold training → evaluation → ONNX → app. |
| `fer_lib.py` | Library code: indexing, de-duplication, cropping, HSEmotion/timm models, training loop, ONNX export, analysis. This code must live in a `.py` module so DataLoader workers can import it on Windows. |
| `fer_infer_onnx.py` | Standalone inference (numpy + OpenCV + onnxruntime). Its `square_crop` is used by the training crops **and** by `detector/face_engine.py`, so training and live crops are identical. It also works as a CLI: `python training/fer_infer_onnx.py --model-dir face_expression_model --image photo.jpg` (or `--folder dir --out-csv preds.csv`, or `--webcam`). |
| `requirements-train.txt` | Training dependencies. Install PyTorch separately (see the top of the file). |

## Notes

* **GitHub size limit:** 5 × B0 gives an ONNX file of about 80 MB, and 5 × B2 is about 150 MB,
  which is over GitHub's 100 MB per-file limit. For large ensembles, use Git LFS
  (`git lfs track "*.onnx"`) or keep the model file out of git.
* **Smoke test:** run `FER_SMOKE_TEST=1 jupyter nbconvert --to notebook --execute face_expression_kfold.ipynb --output smoke_run.ipynb`.
  It uses a tiny subset with 3 folds × 2 epochs and writes to separate folders, so it checks the
  code end to end without touching your real runs.
