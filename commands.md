```bash
conda activate E:\computer_science\research\knnmt\paec\venv
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r E:\computer_science\research\knnmt\paec\requirements.txt

pip list --format=freeze | findstr "torch transformers faiss uvicorn"  # Windows
conda list  # 查看所有包（含pip安装的）
python -c "import torch; print(torch.__version__)"  # 测试PyTorch
```

```bash
python scripts/01_generate_training_data.py # 生成数据

python scripts/02_run_scientific_experiments.py # 复现实验分析
```

```bash
rm -rf ~/.cache/*
rm -rf ~/.local/share/Trash/*
python -m spacy download de_core_news_lg
python -m spacy download en_core_web_trf
```

## Special
### reimagined-waffle: 0.05, 0.30
### special-barnacle: 0.10, 0.35
### reimagined-meme: 0.15, 0.40
### animated-carnival: 0.20, 0.45
### Local: 0.25, 0.50

