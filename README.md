
<div align= "center">
    <h1>Code For Our Papaer</h1>
</div>

## 🛠️1. Setup
### 🔍 1.1  LLM Environment
We use LLAMA-3-8B as the high-level semantic planner. Please install it via Ollama.

### 🐍 1.2 Python Environment
Create and activate a Python environment:
```
conda create -n LERL python=3.10.8
conda activate LERL
pip install torch==1.13
pip install numpy==2.3.2
pip install pandas==1.23.5
```

## 📁 2. Download the Data
You can manually download the compressed dataset from the following link:
🔗 [Download from Google Drive](https://drive.google.com/file/d/17VLPcdqYpOt2maqvO1p-TFzaRpss-UjZ/view?usp=sharing)


## 🚀 3. Running the Code
### 🧠 3.1  Launching LLM Instances
Start three concurrent LLAMA-3-8B servers (each in the background):
```
nohup python run_llama/run_llama1.py &
nohup python run_llama/run_llama2.py &
nohup python run_llama/run_llama3.py &
```

### 👤 3.2 Training the User Model
On KuaiRand:
```
cd code
bash scripts/run_multibehavior.sh
```
On KuaiRec:
```
bash scripts/run_multibehavior_rec.sh
```

### 🎯 3.3 Running the  Policy
On KuaiRand：
```
bash scripts/train_lerlc.sh
```
On KuaiRec：

```
bash scripts/train_lerlc_rec.sh
```

