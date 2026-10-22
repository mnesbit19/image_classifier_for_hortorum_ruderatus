# Bombus ruderatus versus Bombus hortorum

An image-classification experiment with two cryptic bumblebee species,
*Bombus ruderatus* and *Bombus hortorum*.

I trained an ImageNet-pretrained EfficientNet-B0 on public iNaturalist photos
to see how far a simple two-species classifier would get. On the held-out test,
it reached **66.9% balanced accuracy** (reported 95% bootstrap interval
61.8–71.3%). It made many false *ruderatus* calls: only **27.4% of those
predictions agreed with the reference labels**.

The reference labels matter here. These are iNaturalist community
identifications, not independently verified species identities. The photos
were checked for file problems and screened for duplicates and suspect images,
but that did not establish that the bees were correctly identified. The results
measure agreement with those labels. They cannot tell us how much of the
disagreement comes from the model and how much comes from errors in the labels.

## What was tested

The selected dataset contained 14,892 photos: 2,416 labelled *ruderatus* and
12,476 labelled *hortorum*. Photographers, observations and known exact
duplicates were kept within one split. There were 12,108 training photos,
1,469 validation photos and 1,315 test photos. The model used full photos resized
and padded to 224 × 224 pixels, without manual bee crops.

Ten training epochs were run. The epoch-3 checkpoint was selected on validation
balanced accuracy, then evaluated once on the held-out test with the original
decision rule. No model or threshold was selected using the test results.

The main results average predictions within observation/exact-duplicate groups,
so multiple views of the same observation do not count as separate examples.

| Test measure | Result |
| --- | ---: |
| Observation/exact groups | 633 |
| Balanced accuracy | 66.9% |
| Overall accuracy | 63.8% |
| Ruderatus recall | 71.4% |
| Ruderatus precision | 27.4% |

Balanced accuracy is the mean recall across the two species. It is useful here
because *hortorum* is much more common in the dataset. It is different from
overall accuracy.

The experiment produced a working training and prediction pipeline, with some
agreement beyond a single-class prediction. It did not establish a reliable
species-identification tool. See [METHODS_RESULTS.md](METHODS_RESULTS.md) for the
confusion matrix, methods, data provenance and limitations.

## Code

- `code/download_inaturalist.py`: photo acquisition and licence records.
- `code/check_images.py`, `code/scan_near_duplicates.py`: technical checks and
  duplicate candidates.
- `code/prepare_baseline.py`: grouped train/validation/test splits.
- `code/train_baseline.py`: EfficientNet-B0 training.
- `code/predict_photo.py`: prediction for one photo using a saved checkpoint.
- `code/evaluate_baseline.py`: the original held-out evaluation and bootstrap.

The scripts and configurations record the experiment as run. Training and
evaluation require the original data and run artifacts; this code release does
not include photographs, model weights or a ready-to-use identification service.
The original one-time test must not be rerun to choose a better model.

Training used Python 3.12.11, PyTorch 2.7.1, torchvision 0.22.1, Pillow 12.3.0
and NumPy 2.2.6. Given a compatible saved checkpoint and those dependencies,
the predictor accepts a new photo:

```bash
python code/predict_photo.py --checkpoint /path/to/best.pt --image /path/to/bee.jpg --output prediction.json
```

It assumes the subject belongs to this species pair. It cannot reject other
species, and its scores are not calibrated identification probabilities.

## Data and licence

Photos came from iNaturalist Research Grade observations. Acquisition stopped
with 2,488 *ruderatus* and 12,638 *hortorum* photos. The *hortorum* download was
budget-limited and remains recorded as **INCOMPLETE**; it was not a random sample
of the full export. Conservative exclusions removed 234 photos before splitting.

Photo licences and attribution were recorded individually. No photographs are
redistributed here. The [GNU GPLv3 licence](LICENSE) applies to the original
project code, not to iNaturalist photographs, third-party data, libraries or
pretrained weights.

## Result record

Training finished on 20 October 2026 and the final test on 21 October 2026.
The saved test predictions, source and output checksums, and completion record
were retrieved and checked on 22 October. Recomputing the metrics and bootstrap
from those saved predictions reproduced the reported results. The public
results folder includes the predictions and summary. To reproduce the numbers
without photographs, weights or PyTorch:

```bash
python tools/reproduce_results.py
```

This reads saved scores; it does not run the model again. The interval does not
account for uncertain species labels.
