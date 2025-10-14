import pandas as pd
import numpy as np
import os
import re
import torch
import lightgbm as lgb
from torchvision import models, transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from typing import List

# --- CONFIGURATION ---
DATASET_FOLDER = 'student_resource/dataset/'
IMAGE_DIR = 'student_resource/images'
TARGET = 'log_price'
TEXT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


# --- 1. FEATURE ENGINEERING FUNCTIONS (from notebook EDA) ---

def extract_brand_v5(text):
    """Extracts brand/item name before the first comma or newline."""
    if pd.isna(text): return "UNKNOWN_BRAND"
    content = str(text).strip()
    # Capture content after 'Item Name:' up to the first comma or newline
    match = re.search(r'Item Name:\s*(.*?)(?:,|$|\n)', content, re.IGNORECASE | re.DOTALL)
    if match:
        name_segment = match.group(1).strip()
        final_name = name_segment
        first_comma_index = final_name.find(',')
        if first_comma_index != -1:
             final_name = final_name[:first_comma_index].strip()
        return final_name.upper() if final_name else "UNKNOWN_BRAND"
    return "UNKNOWN_BRAND"

def extract_size_and_unit_v2(text):
    """Extracts numerical size and categorical unit."""
    if pd.isna(text): return np.nan, "NO_UNIT"
    # Robust pattern for decimal numbers followed by common units
    pattern = re.compile(
        r'((\d+\.?\d*)|(\.\d+))\s*(ounce|fl oz|ct|pound|lb|count|pack|gallon|liter)s?', 
        re.IGNORECASE
    )
    match = pattern.search(str(text))
    if match:
        try:
            value = float(match.group(1))
        except ValueError:
            return np.nan, "NO_UNIT"
        unit = match.group(4).upper()
        return value, unit
    return np.nan, "NO_UNIT"

def extract_ipq(text):
    """Extracts Item Pack Quantity (IPQ)."""
    if pd.isna(text): return 1
    patterns = [r'(\d+)\s*[-\s]?\s*(?:pack|count|set|quantity|dozen)s?', 
                r'(?:pack|count|set|quantity|dozen)s?\s*of\s*(\d+)',
                r'x\s*(\d+)']
    for pattern in patterns:
        match = re.search(pattern, str(text), re.IGNORECASE)
        if match: return int(match.group(1)) 
    return 1 

def standardize_size(row):
    """Converts various units into a standard ounce/fl oz value."""
    value = row['size_value']
    unit = row['size_unit']
    if pd.isna(value) or unit == 'NO_UNIT': return np.nan
    if unit == 'FL OZ' or unit == 'LITER' or unit == 'GALLON': # Volume conversion
        if unit == 'LITER': value *= 33.814
        if unit == 'GALLON': value *= 128.0
        return value
    if unit == 'OUNCE' or unit == 'LB' or unit == 'POUND': # Weight conversion
        if unit == 'LB' or unit == 'POUND': value *= 16.0
        return value
    return np.nan

# --- 2. MODEL AND DEVICE SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ResNet Setup
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
feature_extractor.eval() 
feature_extractor.to(device) 
preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# MiniLM Setup
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME)
text_model.to(device)


# --- 3. EMBEDDING FUNCTIONS (GPU Optimized) ---

def extract_single_image_feature(sample_id, image_dir=IMAGE_DIR):
    """Loads and extracts a 2048-dim feature vector using ResNet-50 on GPU."""
    filename = os.path.join(image_dir, f"{sample_id}.jpg")
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return np.zeros(2048, dtype=np.float32) 
    try:
        img = Image.open(filename).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = feature_extractor(img_tensor).squeeze().cpu().numpy()
            return features
    except Exception:
        return np.zeros(2048, dtype=np.float32) 

def get_image_features_df(df: pd.DataFrame) -> pd.DataFrame:
    features_list = []
    # Process images for the given DataFrame (Train or Test)
    for index, row in df.iterrows():
        features = extract_single_image_feature(row['sample_id'])
        features_list.append(features)
    
    image_features = np.array(features_list)
    embedding_col_names = [f'image_embed_{i}' for i in range(image_features.shape[1])]
    return pd.DataFrame(image_features, columns=embedding_col_names, index=df.index)

def get_text_embeddings_df(df: pd.DataFrame) -> pd.DataFrame:
    BATCH_SIZE = 128
    texts = df['catalog_content'].fillna('').tolist()
    all_embeddings = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        # Encode and move input tensor to GPU
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
        
        with torch.no_grad():
            model_output = text_model(**encoded_input)
            # Mean Pooling and move result back to CPU
            embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embeddings)

    text_features = np.concatenate(all_embeddings, axis=0)
    embedding_dim = text_features.shape[1] 
    embedding_col_names = [f'text_embed_{j}' for j in range(embedding_dim)]
    return pd.DataFrame(text_features, columns=embedding_col_names, index=df.index)


# --- 4. MAIN EXECUTION PIPELINE ---
def run_pipeline_and_predict():
    print("Starting Multimodal Pricing Pipeline...")
    
    # Load Data
    train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # 1. Base Feature Creation
    for df in [train, test]:
        df['log_price'] = np.log1p(df['price']) if 'price' in df.columns else np.nan
        df['IPQ'] = df['catalog_content'].apply(extract_ipq)
        df['word_count'] = df['catalog_content'].apply(lambda x: len(str(x).split()))
        df['char_count'] = df['catalog_content'].apply(lambda x: len(str(x)))
        df['brand_name'] = df['catalog_content'].apply(extract_brand_v5)
        df[['size_value', 'size_unit']] = df['catalog_content'].apply(lambda x: pd.Series(extract_size_and_unit_v2(x)))
        df['standard_size_oz'] = df.apply(standardize_size, axis=1)

    # 2. Target Encoding Setup (Critical: Training data only)
    BRAND_THRESHOLD = 10
    unit_threshold = 10
    
    # Group rare categories
    brand_counts = train['brand_name'].value_counts()
    rare_brands = brand_counts[brand_counts < BRAND_THRESHOLD].index
    brand_map = {brand: 'OTHER_BRAND' for brand in rare_brands}
    unit_counts = train['size_unit'].value_counts()
    rare_units = unit_counts[unit_counts < unit_threshold].index
    unit_map = {unit: 'OTHER_UNIT' for unit in rare_units}
    
    for df in [train, test]:
        df['brand_name_grouped'] = df['brand_name'].replace(brand_map)
        df['size_unit_grouped'] = df['size_unit'].replace(unit_map)
        
    # Apply Target Encoding
    brand_mean_prices = train.groupby('brand_name_grouped')['log_price'].mean()
    unit_mean_prices = train.groupby('size_unit_grouped')['log_price'].mean()
    
    train['brand_encoded'] = train['brand_name_grouped'].map(brand_mean_prices)
    test['brand_encoded'] = test['brand_name_grouped'].map(brand_mean_prices)
    train['unit_encoded'] = train['size_unit_grouped'].map(unit_mean_prices)
    test['unit_encoded'] = test['size_unit_grouped'].map(unit_mean_prices)
    
    print("Feature Engineering and Encoding Complete.")
    
    # 3. Embedding Extraction (GPU-Accelerated)
    train = train.join(get_image_features_df(train))
    test = test.join(get_image_features_df(test))
    train = train.join(get_text_embeddings_df(train))
    test = test.join(get_text_embeddings_df(test))
    print(f"Total Embeddings Extracted: {len([c for c in train.columns if 'embed_' in c])}")

    # 4. Final Feature Selection and Imputation
    TEXT_EMBED_COLS = [col for col in train.columns if 'text_embed_' in col]
    IMAGE_EMBED_COLS = [col for col in train.columns if 'image_embed_' in col]
    NUMERICAL_FEATS = ['standard_size_oz', 'word_count', 'char_count', 'IPQ'] 
    ENCODED_FEATS = ['brand_encoded', 'unit_encoded'] 
    
    FINAL_FEATURES_FULL = NUMERICAL_FEATS + ENCODED_FEATS + TEXT_EMBED_COLS + IMAGE_EMBED_COLS
    
    X = train[FINAL_FEATURES_FULL].copy()
    y = train[TARGET]
    X_test = test[FINAL_FEATURES_FULL].copy()

    # Imputation: Use train median for both
    imputation_values = X.median()
    X = X.fillna(imputation_values)
    X_test = X_test.fillna(imputation_values)
    
    # 5. Model Training and Submission
    final_lgbm = lgb.LGBMRegressor(
        random_state=42, metric='mae', n_estimators=2000, learning_rate=0.03, 
        n_jobs=-1, verbose=-1, device='gpu', gpu_platform_id=0, gpu_device_id=0
    )
    
    final_lgbm.fit(X, y)

    # Predict and Save
    log_test_predictions = final_lgbm.predict(X_test)
    final_price_predictions = np.expm1(log_test_predictions)
    
    submission_df = test[['sample_id']].copy()
    submission_df['price'] = np.maximum(0.01, final_price_predictions)
    submission_df.to_csv(os.path.join(DATASET_FOLDER, 'test_out.csv'), index=False)
    print("Submission file successfully regenerated.")


if __name__ == "__main__":
    run_pipeline_and_predict()