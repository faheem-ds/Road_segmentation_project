# Road Segmentation Project

Deep learning project for binary road segmentation from satellite/aerial imagery.

The task is semantic segmentation:

- background = `0`
- road = `1`

The project compares three models using the same dataset split, preprocessing, metrics, and output format:

1. U-Net CNN baseline
2. SegFormer Transformer baseline
3. Custom Hybrid model, `EdgeAwareHybridRoadFormer`

## Current Status

The project is configured for a practical training subset before full final training:

```python
TRAIN_SAMPLES = 1200
VALID_SAMPLES = 150
TEST_SAMPLES = 150
IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_EPOCHS = 100
THRESHOLD = 0.45
```

The prepared full split contains approximately:

```text
train: 4669 images
valid: 933 images
test: 624 images
```

Use `TRAIN_SAMPLES = None`, `VALID_SAMPLES = None`, and `TEST_SAMPLES = None` if you want to train/evaluate on all available images.

## Install

```bash
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build and verify it:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Dataset

The dataset path is controlled in `config.py`:

```python
DATASET_DIR = PROJECT_DIR / "dataset_prepared"
```

The prepared dataset folder is not tracked in git. Download `dataset_prepared/` from:

[Google Drive dataset/checkpoints folder](https://drive.google.com/drive/folders/1U_DdDzq9xT2sOnpWn6DIYR2xeWI2v1dz)

Expected layout:

```text
dataset_prepared/
  train/images/
  train/masks/
  valid/images/
  valid/masks/
  test/images/
  test/masks/
```

Images and masks are matched by filename key, for example:

```text
100034_sat.jpg -> 100034_mask.png
```

To recreate the split:

```bash
python prepare_splits.py
```

## Preprocessing And Augmentation

Images are resized to `IMAGE_SIZE x IMAGE_SIZE` with bilinear interpolation. Masks are resized with nearest-neighbor interpolation and converted to binary masks.

Training augmentation is enabled with:

```python
AUGMENT_TRAIN = True
```

The dataset pipeline uses road-friendly augmentations such as flips, rotations, brightness/contrast changes, shift/scale, and crop/resize style transforms. Images are normalized with ImageNet mean and standard deviation.

Important: sigmoid is used only during evaluation/inference. Training losses use raw logits.

## Configuration

Main settings live in `config.py`:

```python
IMAGE_SIZE = 512
BATCH_SIZE = 8
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

LR_UNET = 1e-4
LR_SEGFORMER = 5e-5
LR_CUSTOM = 5e-5
LR_BACKBONE = 1e-5
FREEZE_BACKBONE_EPOCHS = 5

LAMBDA_EDGE = 0.2
LAMBDA_CENTERLINE = 0.1
LAMBDA_CLDICE = 0.05
THRESHOLD = 0.45

CUSTOM_BACKBONE = "internimage_tiny"
D_MODEL = 256
NHEAD = 8
NUM_DECODER_LAYERS = 3
NUM_QUERIES = 4
```

Post-processing is controlled with:

```python
APPLY_POSTPROCESSING = True
POSTPROCESS_MIN_COMPONENT_SIZE = 32
POSTPROCESS_CLOSING_KERNEL = 5
POSTPROCESS_CLOSING_ITERATIONS = 1
```

## Train Models

Train U-Net:

```bash
python train_unet.py
```

Train SegFormer:

```bash
python train_segformer.py
```

Train the custom hybrid model:

```bash
python train_custom.py
```

Train all three models one by one without saving test prediction images:

```bash
python train_all.py
```

This is useful for final training because large test visual outputs can take extra time and disk space. After training, run comparison separately on a smaller test subset.

## Resume Custom Training

The custom model saves:

```text
checkpoints/custom_hybrid_best.pth
checkpoints/custom_hybrid_last.pth
```

The `checkpoints/` folder is not tracked in git. Download available trained checkpoints from:

[Google Drive dataset/checkpoints folder](https://drive.google.com/drive/folders/1U_DdDzq9xT2sOnpWn6DIYR2xeWI2v1dz)

To resume from the last saved epoch:

```bash
python train_custom.py --resume
```

To train or resume without final test image saving:

```bash
python train_custom.py --resume --no-test
```

Resume works when the model architecture and important config values match the saved checkpoint.

## Model Details

### U-Net

The U-Net baseline uses `segmentation_models_pytorch` with a pretrained ResNet34 encoder when available.

Checkpoint:

```text
checkpoints/best_unet.pth
```

Outputs:

```text
outputs/unet/
```

### SegFormer

The SegFormer baseline uses a Cityscapes-pretrained model:

```python
SEGFORMER_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
```

The original Cityscapes classifier has 19 classes, so the final segmentation classifier is reinitialized for one binary road channel. This warning is expected.

Checkpoint:

```text
checkpoints/best_segformer.pth
```

Outputs:

```text
outputs/segformer/
```

### Custom Hybrid

The custom model is defined in:

```text
models/custom_hybrid.py
```

Model name:

```text
EdgeAwareHybridRoadFormer
```

The design is a practical InternImage + Mask2Former-inspired road model:

- InternImage-style backbone attempt
- timm fallback backbones such as ConvNeXt-Tiny, ResNet34, or Swin-Tiny
- lightweight local CNN fallback if timm/backbone loading fails
- multi-scale FPN projection into transformer tokens
- compact road queries
- cross-attention semantic decoder
- fine-feature CNN refinement decoder for denser road masks
- segmentation head
- edge head
- centerline/connectivity head

The model returns:

```python
{
    "seg_logits": seg_logits,
    "edge_logits": edge_logits,
    "centerline_logits": centerline_logits,
}
```

The custom trainer freezes the pretrained backbone for the first `FREEZE_BACKBONE_EPOCHS`, then unfreezes it with differential learning rates:

- lower learning rate for the backbone
- higher learning rate for decoder/refinement heads
- cosine learning rate schedule after unfreezing

Checkpoint:

```text
checkpoints/custom_hybrid_best.pth
```

Outputs:

```text
outputs/custom_hybrid/
```

## Loss Functions

U-Net and SegFormer use:

```text
BCEWithLogitsLoss + Dice Loss
```

The custom hybrid model uses:

```text
total_loss =
    segmentation_loss
  + LAMBDA_EDGE * edge_loss
  + LAMBDA_CLDICE * soft_clDice_loss
  + LAMBDA_CENTERLINE * centerline_loss
```

Where:

- `segmentation_loss` is BCE with logits plus Dice loss
- `edge_loss` compares predicted edges with generated edge labels
- `soft_clDice_loss` encourages connected thin road structures
- `centerline_loss` encourages the model to learn road centerlines/skeletons

Edge labels are generated from masks using a PyTorch morphological gradient:

```text
edge = max_pool(mask) - min_pool(mask)
```

## Metrics

All models are evaluated with the same main segmentation metrics:

- IoU
- Dice Score
- Precision
- Recall
- Pixel Accuracy

Additional road-structure metrics:

- Edge IoU
- Connected Component Difference
- Largest Component Ratio

Connected Component Difference is:

```text
abs(predicted_road_components - ground_truth_road_components)
```

Largest Component Ratio is:

```text
largest predicted road component area / total predicted road area
```

Lower component difference is better. Higher largest component ratio often means the predicted road network is more connected.

## Compare Models

After training all models:

```bash
python compare_models.py
```

This saves:

```text
outputs/comparison_results.csv
outputs/comparisons/
```

The comparison table contains:

```text
Model | IoU | Dice | Precision | Recall | Pixel Accuracy | Edge IoU | Component Diff | Largest Component Ratio
```

Combined visualizations show:

```text
original image | ground truth | U-Net prediction | SegFormer prediction | Custom Hybrid prediction
```

## Evaluate One Model

```bash
python evaluate.py --model unet
python evaluate.py --model segformer
python evaluate.py --model custom_hybrid
```

With a custom checkpoint:

```bash
python evaluate.py --model custom_hybrid --checkpoint checkpoints/custom_hybrid_best.pth
```

## Run On External Test Images

Put images in:

```text
test_image/
```

Then run:

```bash
python infer_test_images.py
```

Outputs are saved to:

```text
outputs/test_image_predictions/
```

You can run only selected models:

```bash
python infer_test_images.py --models unet custom_hybrid
```

Disable post-processing:

```bash
python infer_test_images.py --no-postprocess
```

## Current Comparison Snapshot

Latest saved comparison in `outputs/comparison_results.csv`:

| Model | IoU | Dice | Precision | Recall | Pixel Accuracy | Edge IoU | Component Diff | Largest Component Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | 0.5988 | 0.7475 | 0.7316 | 0.7689 | 0.9785 | 0.7608 | 3.9825 | 0.7880 |
| SegFormer | 0.5541 | 0.7113 | 0.6669 | 0.7693 | 0.9742 | 0.6861 | 5.3136 | 0.7420 |
| Custom Hybrid | 0.4740 | 0.6411 | 0.6454 | 0.6418 | 0.9703 | 0.6235 | 17.7675 | 0.4704 |

At this checkpoint, U-Net is still the strongest baseline. The custom model has the research additions: cross-attention decoder, edge-aware loss, centerline head, soft clDice, and connectivity-aware post-processing, but it needs longer or more stable training to compete with the pretrained baselines.

For the final report, retrain all three models under the same data subset/full-data setting and then run `compare_models.py` once using the same `THRESHOLD`.

## Practical Notes

- U-Net often trains fastest and gives strong early results.
- SegFormer is pretrained on Cityscapes, which is closer to roads/street scenes than ADE-style general segmentation.
- The custom hybrid model is slower because it uses a CNN backbone plus transformer attention plus auxiliary heads.
- If custom training is too slow, use `--no-test` during training and run comparison later.
- For an 8GB GPU, reduce `BATCH_SIZE` if CUDA runs out of memory.
- If the custom model is interrupted, resume with `python train_custom.py --resume`.
