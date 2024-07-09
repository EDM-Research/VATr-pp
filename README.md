# Handwritten Text Generation from Visual Archetypes ++

This repository includes the code for training the VATr++ Styled Handwritten Text Generation model.

## Installation

```bash
conda create --name vatr python=3.9
conda activate vatr
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
git clone https://github.com/aimagelab/VATr.git && cd VATr
pip install -r requirements.txt
```

[This folder](https://drive.google.com/drive/folders/13rJhjl7VsyiXlPTBvnp1EKkKEhckLalr?usp=sharing) contains the regular IAM dataset `IAM-32.pickle` and the modified version with attached punctuation marks `IAM-32-pa.pickle`.
The folder also contains the synthetically pretrained weights for the encoder `resnet_18_pretrained.pth`.
Please download these files and place them into the `files` folder.

## Training

To train the regular VATr model, use the following command. This uses the default settings from the paper.

```bash
python train.py
```

Useful arguments:
```bash
python train.py
        --feat_model_path PATH  # path to the pretrained resnet 18 checkpoint. By default this is the synthetically pretrained model
        --is_cycle              # use style cycle loss for training
        --dataset DATASET       # dataset to use. Default IAM
        --resume                # resume training from the last checkpoint with the same name
        --wandb                 # use wandb for logging
```

Use the following arguments to apply full VATr++ training
```bash
python train.py
        --d-crop-size 64 128          # Randomly crop input to discriminator to width 64 to 128
        --text-augment-strength 0.4   # Text augmentation for adding more rare characters
        --file-suffix pa              # Use the punctuation attached version of IAM
        --augment-ocr                 # Augment the real images used to train the OCR model
```

### Pretraining dataset
The model `resnet_18_pretrained.pth` was pretrained by using this dataset: [Font Square](https://github.com/aimagelab/font_square)


## Generate Styled Handwritten Text Images

We added some utility to generate handwritten text images using the trained model. These are used as follows:

```bash
python generate.py [ACTION] --checkpoint files/vatrpp.pth
```

The following actions are available with their respective arguments.

### Custom Author

Generate the given text for a custom author.

```bash
text  --text STRING     # String to generate
      --text-path PATH  # Optional path to text file
      --output PATH     # Optional output location, default: files/output.png
      --style-folder PATH    # Optional style folder containing writer samples, default: 'files/style_samples/00'
```
Style samples for the author are needed. These can be automatically generated from an image of a page using `create_style_sample.py`.
```bash
python create_style_sample.py  --input-image PATH     # Path of the image to extract the style samples from.
                               --output-folder PATH   # Folder where the style samples should be saved
```

### All Authors

Generate some text for all authors of IAM. The output is saved to `saved_images/author_samples/`

```bash
authors --test-set        # Generate authors of test set, otherwise training set is generated
        --checkpoint PATH # Checkpoint used to generate text, files/vatr.pth by default
        --align           # Detect the bottom lines for each word and align them
        --at-once         # Generate the whole sentence at once instead of word-by-word
        --output-style    # Also save the style images used to generate the words
```

### Evaluation Images

```bash
fid --target_dataset_path PATH  # dataset file for which the test set will be generated
    --dataset-path PATH         # dataset file from which style samples will be taken, for example the attached punctuation
    --output PATH               # where to save the images, default is saved_images/fid
    --checkpoint PATH           # Checkpoint used to generate text, files/vatr.pth by default
    --all-epochs                # Generate evaluation images for all saved epochs available (checkpoint has to be a folder)
    --fake-only                 # Only output fake images, no ground truth
    --test-only                 # Only generate test set, not train set
    --long-tail                 # Only generate words containing long tail characters
```