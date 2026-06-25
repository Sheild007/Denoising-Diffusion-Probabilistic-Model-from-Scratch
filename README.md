# Diffusion Model — How to Run

## Install

```bash
pip install torch torchvision pillow matplotlib
```

## Dataset

Place the data outside this folder with the layout:

```
animal_data/<ClassName>/*.jpg
```

## Train

```bash
python BSCS22008_05.py train --data_dir /path/to/animal_data
```

## Sample

```bash
python BSCS22008_05.py sample --checkpoint saved_models/diffusion_model.pt --num_samples 4
```

Generated images are written to `results/`.
