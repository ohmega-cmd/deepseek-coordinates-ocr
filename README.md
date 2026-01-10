# DeepSeek-Coordinates-OCR
OCR baseado no DeepSeek para reconhecimento e extração automática de coordenadas geográficas a partir de imagens.

DeepSeek-based OCR for automatic recognition and extraction of geographic coordinates from images.

## Install

>Our environment is cuda11.8+torch2.6.0.
1. Clone this repository and navigate to the DeepSeek-OCR folder
```bash
git clone https://github.com/ohmega-cmd/deepseek-coordinates-ocr.git
```
2. On root folder clone this repository
```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
```
2. Conda
```Shell
conda create -n deepseek-ocr python=3.10.19 -y
conda activate deepseek-ocr
```
3. Packages
```Shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \ --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.

This repository uses DeepSeek-OCR as an external dependency https://github.com/deepseek-ai/DeepSeek-OCR.git.
All rights related to DeepSeek belong to their respective authors.
