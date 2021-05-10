# OcrWeb
### 1. Result
![ocrweb](https://user-images.githubusercontent.com/30888482/116226251-ba22c400-a78d-11eb-8cfa-4a23baaafccc.PNG)

### 2. Usage
python train_rough - train with existed dataset <br>
python main.py <br>
http://127.0.0.1:5000 <br>
image upload <br>
python train_fine - fine tuning <br>

### 3. Acknowlegements
this implementation has been tested with 64bit Python 3.8.6 and pytorch 1.8.1+cu111

### 4. References
https://towardsdatascience.com/pytorch-scene-text-detection-and-recognition-by-craft-and-a-four-stage-network-ec814d39db05 <br>
https://github.com/clovaai/CRAFT-pytorch (CRAFT pytorch)<br>
https://github.com/clovaai/deep-text-recognition-benchmark (recognition)<br>
https://www.robots.ox.ac.uk/~vgg/data/text/ (text recognition dataset - English)
https://www.aihub.or.kr/aidata/133 (text recognition dataset - Korean)
