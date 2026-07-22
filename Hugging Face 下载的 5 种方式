# Hugging Face 下载的 5 种方式
## 方式 1：Git Clone
git clone https://hf-mirror.com/BAAI/bge-m3
cd bge-m3
git lfs pull  # 确保大文件完整

## 方式 2：huggingface-cli / hf 命令
hf download BAAI/bge-m3 --local-dir ./bge-m3

### 下载数据集
hf download squad --repo-type dataset --local-dir ./squad

## 方式 3：Python 代码（snapshot_download）
from huggingface_hub import snapshot_download

# 下载模型
snapshot_download(repo_id='BAAI/bge-m3', local_dir='./bge-m3')

# 下载数据集
snapshot_download(repo_id='squad', repo_type='dataset', local_dir='./squad')

## 方式 4：transformers / datasets 库（自动加载）
# 模型自动下载
from transformers import AutoModel
model = AutoModel.from_pretrained('BAAI/bge-m3')

# 数据集自动下载
from datasets import load_dataset
dataset = load_dataset('squad')

自动缓存到 ~/.cache，不易管理

## 方式 5：wget / curl（手动下载单文件）
# 下载单个文件
wget https://hf-mirror.com/BAAI/bge-m3/resolve/main/config.json
wget https://hf-mirror.com/BAAI/bge-m3/resolve/main/model.safetensors

需要手动下载所有文件
不知道完整文件列表
