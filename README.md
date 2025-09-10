# 🌌 Galaxy Classification using Deep Learning

## 📌 Overview
This project implements a deep learning approach to classify galaxies into five morphological categories:  
- **Irregular**  
- **Spiral**  
- **Elliptical**  
- **Lenticular**  
- **Peculiar**  

The model is trained on grayscale images resized to a fixed size and evaluated on a test dataset.  
It serves as a **baseline implementation** using a Dense Neural Network (MLP).  
Future improvements may include CNN architectures for better performance.  

---

## 📂 Dataset
- **Input**: Grayscale galaxy images  
- **Preprocessing**:  
  - Resized to **150×150**  
  - Normalized to values in `[0, 1]`  
- **Categories**: `Irregular, Spiral, Elliptical, Lenticular, Peculiar`  

> ⚠️ Adjust the dataset path inside the script:  
```python
db_path = '/path/to/dataset'
