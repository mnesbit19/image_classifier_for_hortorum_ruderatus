# Methods and results

Miles Nesbit · 22 October 2026

This experiment asked whether a standard image classifier could distinguish
photos labelled *Bombus ruderatus* from photos labelled *Bombus hortorum*.
It was a small proof of concept using community-labelled photographs. There
was no independent genetic or expert reference set for this experiment.

## Photos and selection

The source was two iNaturalist Research Grade exports, 795893 (*ruderatus*)
and 797562 (*hortorum*). Eligible photos were downloaded at original size,
with observation IDs, source labels, photo licences, attribution and byte
hashes retained. The photo licence allowlist was CC0, CC BY and CC BY-NC.
Observation licences were not treated as substitutes for photo licences.

| Cohort | Accepted source photos | Photos used after exclusions |
| --- | ---: | ---: |
| Ruderatus | 2,488 | 2,416 |
| Hortorum | 12,638 | 12,476 |
| Total | 15,126 | 14,892 |

The *ruderatus* acquisition completed. The *hortorum* acquisition stopped at
the configured download budget and its source status remains **INCOMPLETE**.
The saved, export-order subset was accepted for this experiment; acquisition
was closed on 16 October 2026. These are not balanced or random samples of the
two species' field populations.

File integrity, image decoding and exact duplicates were checked. A perceptual
similarity scan produced candidates for visual screening. Before splitting,
234 previously flagged or reviewed suspect photos were excluded, including
photos from the implicated cross-observation duplicate groups. Raw originals
were retained. There was no blanket exclusion of every similarity candidate,
no relabelling based on model output, and no manual cropping of the dataset.

These checks addressed technical problems, duplication and some visibly
unsuitable images. They were not a systematic expert verification of species
identity. The retained community labels may still contain mistakes, but their
error rate is unknown.

## Splits and model

Seed 42 was used to assign connected photographer/observation/known-duplicate
components to one split. Known shared photo IDs, byte duplicates and exact
pixel duplicates do not cross splits. Unrecorded repeat individuals could
remain, and geographic independence was not established.

| Split | Hortorum | Ruderatus | Total photos |
| --- | ---: | ---: | ---: |
| Training | 10,159 | 1,949 | 12,108 |
| Validation | 1,250 | 219 | 1,469 |
| Test | 1,067 | 248 | 1,315 |

An ImageNet-pretrained EfficientNet-B0 was trained on full frames. Images were
EXIF-oriented, converted to RGB, resized with bicubic interpolation while
preserving aspect ratio, and padded to 224 × 224 pixels before ImageNet
normalisation. Training used mild horizontal-flip and colour augmentation,
class/group-balanced replacement sampling, batch size 32 and AdamW with
weight decay 0.0001. Three classifier-head epochs at learning rate 0.001 were
followed by seven fine-tuning epochs at 0.0001.

The epoch-3 checkpoint had the highest validation group balanced accuracy
(73.12%) and was retained. The final test used this checkpoint and the fixed
two-output argmax rule, with ties assigned to *hortorum*. No test-based model,
threshold or preprocessing selection was made.

A validation-only runtime check found small differences between GPU training
predictions and CPU inference: the largest was 0.0002882481 across eight fixed
validation photos, with no changed class calls. The cross-device numerical
tolerance was changed from 0.0001 to 0.001 before test release. The species
decision threshold stayed at 0.5. The standalone predictor also passed the
development-photo comparison. This was not an independent field trial.

## Held-out results

Primary predictions average the *ruderatus* score within each observation/exact
group. There were 633 such groups from 635 observations, in 333 split
components. Some evaluation groups therefore share a photographer within the
test split. The reported 95% intervals use 2,000 unstratified percentile
bootstrap draws of whole split components, seed 42.

| Measure | Observation/exact groups | Individual photos |
| --- | ---: | ---: |
| Number | 633 | 1,315 |
| Balanced accuracy | 66.87% | 64.95% |
| Overall accuracy | 63.82% | 62.97% |
| Hortorum recall | 62.31% | 61.76% |
| Ruderatus recall | 71.43% | 68.15% |
| Ruderatus precision | 27.37% | 29.29% |

Group confusion matrix; rows are the iNaturalist reference labels:

| Reference label | Predicted hortorum | Predicted ruderatus |
| --- | ---: | ---: |
| Hortorum | 329 | 199 |
| Ruderatus | 30 | 75 |

The primary balanced accuracy was 66.8696%, with a reported 95% interval of
61.7654–71.3112%. Of 105 groups labelled *ruderatus*, 75 were called correctly.
However, 199 groups labelled *hortorum* were also called *ruderatus*. Only 75
of the model's 274 *ruderatus* calls agreed with the reference label.

Always predicting *hortorum* would give 83.41% overall accuracy on these groups,
but 50% balanced accuracy and no *ruderatus* detections. This explains both the
use of balanced accuracy and why the headline result is not overall accuracy.

## Interpretation

The model learned a pattern that agreed with the held-out community labels
better in balanced accuracy than a single-class prediction. That pattern
could include bee appearance, backgrounds or other features correlated with
the labels; this experiment did not separate them. The large number of false
*ruderatus* calls limits its practical use.

The main unresolved issue is the reference identification. Because these bees
are difficult to identify from photographs, disagreement cannot automatically
be attributed to the model. Equally, it cannot be used to declare the community
label wrong. Without an independently verified reference set, this result does
not estimate accuracy against true species identity. The bootstrap interval
describes resampling variation in this test set, not uncertainty in its labels.

The experiment does not establish that still images cannot work, that video
would solve the problem, or that a different model would necessarily perform
better. It reports this baseline as run. No further training or tuning is part
of this release.

## Evidence and reproducibility

Training job 4420513.pbs-7 completed ten epochs in 02:09:16 on an NVIDIA L40S.
Final-test job 4429740.pbs-7 completed on CPU in 00:07:53, with scheduler,
application and wrapper exit status 0. The selected checkpoint SHA-256 is:

```text
16df795738b138e2414b1ef96deb31e83d26cb657119adb76d37742affc820f9
```

Training artifacts and saved validation predictions were checked locally.
On 22 October the saved final-test run and release record were also retrieved:
18 files, 190,426 bytes. Output checksums and the actual submitted-source
manifest match their saved records. Photo identities, order and group
assignments match the frozen split. Recomputing the metrics, log loss and
all bootstrap intervals from the saved probabilities reproduces the completed
job summary. The release/completion records identify the same fixed checkpoint
and successful test run. No model was run during these checks.

The public results folder contains the original photo/group predictions and
summary, plus a verification record. `python tools/reproduce_results.py`
recomputes the metrics and bootstrap from those scores using Python's standard
library. It does not require photographs, model weights or PyTorch.

The source and frozen configurations are provided to document the method.
Photographs and trained weights are not included. Running the historical
training/evaluation scripts requires the original manifests, input bytes and
training artifacts; the public folder is not a complete rerunnable data bundle.
The original test is complete and is not to be reopened for model selection.
