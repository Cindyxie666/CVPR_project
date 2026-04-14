# Anonymised Facial Expression Recognition

This project is a course group project for `COMP4026 Computer Vision and Pattern Recognition`.

The goal is to anonymise face images while preserving facial expressions, so that:

- face identity becomes difficult for a machine to recognise
- expression recognition performance is preserved as much as possible

The project contains three connected parts:

- `face recognition`: ArcFace-style identity model on Pins Face
- `face anonymization`: U-Net-based anonymizer trained on CelebA-HQ
- `expression recognition`: ResNet-18-based expression classifier on FER-2013

## Project Structure

```text
project/
├─ face_anonymization/
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ generate_anonymized.py
│  ├─ train_arcface.py
│  ├─ eval_arcface.py
│  ├─ dataset.py
│  ├─ config.py
│  └─ models/
├─ expression_recognition/
│  └─ expression_recognition/
│     ├─ train.py
│     └─ eval_expression_consistency.py
├─ COMP4026 Group Project.pdf
└─ README.md
```

## Datasets

The project is designed around the datasets required by the assignment:

- `CelebA-HQ`: training the anonymization model
- `Pins Face Recognition`: training and evaluating the face recognition model
- `FER-2013`: training and evaluating the expression recognition model

Large datasets, checkpoints and generated outputs are ignored by `.gitignore` and should be prepared locally.

## Environment

Install the dependencies for the anonymization module:

```bash
cd face_anonymization
pip install -r requirements.txt
```

The expression recognition module additionally uses:

```bash
pip install scikit-learn seaborn
```

## Training and Evaluation

### 1. Train the expression recognition model

The expression model now follows a more rigorous split:

- `train/val` are created from `data/train`
- `val` is used to select `best_model.pth`
- `data/test` is used only once for final testing

Run:

```bash
cd expression_recognition/expression_recognition
python train.py
```

This script saves:

- `best_model.pth`
- `confusion_matrix.png`
- `training_curves.png`

### 2. Evaluate expression consistency on the hold-out test set

```bash
cd expression_recognition/expression_recognition
python eval_expression_consistency.py --orig_dir data/test --anonymised_dir data/anonymised_test --weights_path best_model.pth
```

This evaluation reports:

- original test accuracy
- anonymised test accuracy
- accuracy drop
- expression consistency

### 3. Train the face recognition model

```bash
cd face_anonymization
python train_arcface.py
```

This trains the ArcFace-style backbone on Pins Face and saves the backbone weights to `pretrained/arcface_r50.pth`.

### 4. Train the face anonymization model

```bash
cd face_anonymization
python train.py
```

The anonymizer uses a composite objective including:

- identity confusion loss
- expression preservation loss
- reconstruction loss
- perceptual loss

### 5. Generate anonymised images

```bash
cd face_anonymization
python generate_anonymized.py --output_dir ../expression_recognition/expression_recognition/data/anonymised_test --grayscale
```

### 6. Run the full anonymization evaluation

```bash
cd face_anonymization
python evaluate.py --checkpoint checkpoints/anonymizer_epoch0010.pth
```

This script evaluates:

- privacy on the Pins Face held-out test set
- utility on the FER-2013 test set

## Current Main Results

Using held-out task-specific test sets:

- face recognition accuracy dropped from `98.3%` to `23.2%`
- expression model achieved `70.7%` best validation accuracy and `69.7%` final FER-2013 test accuracy
- expression accuracy changed from `69.70%` to `69.42%` after anonymization
- expression accuracy drop was only `0.28%`
- expression consistency reached `91.91%`

These results show a useful privacy-utility trade-off:

- identity information is significantly reduced after anonymization
- expression information is largely preserved, with only a very small drop in expression recognition accuracy

## Notes

- The formal final evaluation should use `Pins Face` and `FER-2013` test sets rather than relying only on checks performed on `CelebA-HQ`.
- Pretrained weights and datasets are not included in this repository.
- Paths in the scripts are relative, so commands should be run from the corresponding module directory unless the script arguments are changed.
