# Amazon-ML-Multimodal-Pricing-LGBM

A multimodal machine learning solution for predicting optimal product prices using image, text, and structured metadata fusion. Built with PyTorch, Hugging Face Transformers, and LightGBM.

## 🎯 Performance Summary

- **Final Accuracy**: 60.7937% Average SMAPE (5-Fold CV)
- **Architecture**: Feature-Level Fusion + **LightGBM Regressor** 
- **Model Scale**: 2054+ fused features from multiple modalities
- **Base Improvement**: ~40% gain from baseline SMAPE

## 🔬 Methodology

### Core Components

| Modality | Processing Pipeline | Features | Technology |
|----------|-------------------|-----------|------------|
| Visual | ResNet-50 Transfer Learning | 2048-dim embeddings | **PyTorch/CUDA** |
| Textual | MiniLM Transformer | 384-dim embeddings | **Hugging Face** |
| Metadata | Feature Engineering | Size/Brand/Units | **Regex + LightGBM** |

### Feature Engineering Pipeline

1. **Text Processing**:
   - MiniLM embeddings for semantic understanding
   - Target encoding for high-cardinality features
   - Statistical text feature extraction

2. **Image Processing**:
   - ResNet-50 backbone (pretrained)
   - GPU-accelerated feature extraction
   - Robust handling of missing/corrupt images

3. **Metadata Engineering**:
   - Item Pack Quantity (IPQ) extraction
   - Unit standardization to fluid ounces
   - Brand name normalization

## 🛠️ Setup & Reproducibility

### Requirements
```
pytorch
transformers
lightgbm
pandas
numpy
pillow
scikit-learn
```

### Project Structure
```
├── src/
│   ├── final_submission_script.py    # End-to-end pipeline
│   └── utils.py                      # Helper functions
├── notebooks/
│   └── eda_and_feature_engineering.ipynb
├── data/
│   ├── train.csv
│   └── test.csv
└── images/                           # Product images
```

### Key Model Parameters
```python
lgbm_params = {
    'n_estimators': 2000,
    'learning_rate': 0.03,
    'device': 'gpu',
    'metric': 'mae',
    'verbose': -1
}
```

## 📝 Citation & License

This project is released under the MIT License. Please cite if you use this in your research:

```bibtex
@misc{amazon-ml-multimodal-pricing,
  author = Ankit Sharma,
  title = {Amazon-ML-Multimodal-Pricing-LGBM},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Ankitshrma25/Amazon-ML-Multimodal-Pricing-LGBM}
}
```