好的，收到您的请求。我已经仔细阅读了您提供的三个文件：`定义文档.md`（理论框架）、`日程安排.md`（研究计划）和 `data_generation.py`（Colab 实现代码）。

根据您的研究目标和代码内容，我将为您设计一个清晰、模块化且符合软件工程最佳实践的项目文件结构。这个结构旨在将您庞大的 Colab 脚本分解为逻辑单元，使其更易于管理、扩展和维护，并为后续的模型训练与理论分析阶段做好准备。

以下是为您规划的项目文件夹 `E:\computer_science\research\knnmt\paec` 的层级结构和内容说明。

-----

### **项目根目录 (`paec/`) 结构**

```
paec/
├── data/                      # 存放原始数据集和生成的数据
│   ├── raw/                   # 存放从网络下载的原始数据集 (e.g., WMT, Multi30k)
│   └── processed/             # 存放由 data_generation.py 生成的最终数据
│       ├── knn_mt_training_data.csv
│       ├── knn_mt_training_data.json
│       └── ...
│
├── docs/                      # 存放您的理论和规划文档
│   ├── 01_definition_framework_v2.md
│   └── 02_research_schedule.md
│
├── libs/                      # 存放第三方库或子项目
│   └── knn-box/               # 您提到会把 knn-box 项目放在这里
│
├── models/                    # 存放训练好的模型
│   ├── performance_models/    # 存放 simulator 使用的预训练成本模型
│   │   ├── model_latency_exact.joblib
│   │   └── ...
│   └── dynamics_model/        # [未来工作] 存放您将要训练的动力学模型 T
│
├── results/                   # 存放实验结果、图表和报告
│   ├── knn_mt_enhanced_analysis.png
│   ├── strategy_comparison_summary.csv
│   └── ...
│
├── src/                       # 项目核心源代码
│   ├── core/                  # 定义核心数据结构
│   │   ├── __init__.py
│   │   └── structs.py
│   │
│   ├── system/                # kNN-MT 系统的核心实现
│   │   ├── __init__.py
│   │   ├── knn_mt.py
│   │   └── decoding.py
│   │
│   ├── simulation/            # 环境模拟与策略
│   │   ├── __init__.py
│   │   ├── constraint_simulator.py
│   │   └── policy.py
│   │
│   ├── data_processing/       # 数据加载与处理
│   │   ├── __init__.py
│   │   └── loader.py
│   │
│   ├── analysis/              # 数据分析与可视化
│   │   ├── __init__.py
│   │   ├── visualization.py
│   │   └── scientific_analysis.py
│   │
│   ├── pipeline/              # 整体流程控制
│   │   ├── __init__.py
│   │   └── data_generation_pipeline.py
│   │
│   └── config.py              # 全局配置
│
├── scripts/                   # 可执行脚本，用于运行实验的不同阶段
│   ├── 01_generate_training_data.py
│   └── 02_run_scientific_experiments.py
│
├── .gitignore                 # Git 忽略文件配置
├── README.md                  # 项目说明文件
└── requirements.txt           # Python 依赖库
```

-----

### **各文件/模块内容详解**

#### 1. `requirements.txt` (必须)

这是您项目的所有 Python 依赖。根据您的代码，内容应如下：

```
# Core ML/DL Libraries
torch
transformers
sentence-transformers
faiss-cpu
scikit-learn
numpy

# Data & Analysis
pandas
matplotlib
seaborn
sacrebleu
joblib
umap-learn

# Dataset Loading
datasets
mtdata
fsspec
huggingface-hub

# For installing knn-box from the libs folder
-e libs/knn-box
```

#### 2. `src/` - 核心源代码

* **`src/config.py`**

  * **用途**: 集中管理所有硬编码的参数和配置，便于修改和实验。
  * **包含内容**:

    * `ACTION_SPACE`: `Action` 对象的列表定义。
    * `MODEL_NAMES`: `Helsinki-NLP/opus-mt-de-en` 等模型名称。
    * `SIMULATOR_PARAMS`: 生产约束模拟器的权重 `w1` 到 `w6`、基线延迟等。
    * `PATH_CONFIG`: 指向 `data/`, `models/`, `results/` 等目录的路径变量。

* **`src/core/structs.py`**

  * **用途**: 定义与您理论文档完全对应的核心数据结构。
  * **包含内容**:

    * `class ErrorStateVector(dataclass)`
    * `class ResourcePressureVector(dataclass)`
    * `class GenerativeContextVector(dataclass)`
    * `class SystemState(dataclass)`
    * `class Action(dataclass)`
    * `class DecodingStrategy(Enum)`

* **`src/system/decoding.py`**

  * **用途**: 封装通用的解码算法逻辑。
  * **包含内容**:

    * `class BeamHypothesis(dataclass)`
    * `class BeamSearchDecoder`

* **`src/system/knn_mt.py`**

  * **用途**: kNN-MT 系统的核心实现，负责翻译和状态计算。
  * **包含内容**:

    * `class kNNMTSystem`: 包含模型加载、FAISS 索引、状态计算函数和翻译主逻辑。

      * `__init__()`
      * `_initialize_indexes()`
      * `project_to_query_embedding()`
      * `compute_error_state()`
      * `perform_knn_retrieval()`
      * `compute_context_state()`
      * `translate_with_knn_beam_search()`
      * `_simple_translation_fallback()`

* **`src/simulation/constraint_simulator.py`**

  * **用途**: 模拟生产环境的资源约束。
  * **包含内容**:

    * `class ProductionConstraintSimulator`

      * `__init__()`
      * `_calculate_theoretical_fixed_memory_mb()`
      * `_map_pressure_to_concurrency()`
      * `simulate_traffic_pattern()`
      * `update_resource_metrics()`
      * `_calculate_action_cost()`
      * `compute_pressure_vector()`

* **`src/simulation/policy.py`**

  * **用途**: 实现描述性策略函数 $\pi\left(\cdot \mid \mathbf{S}_t\right)$。
  * **包含内容**:

    * `class DescriptivePolicyFunction`

      * `__init__()`
      * `compute_action_probabilities()`
      * `sample_action()`

* **`src/data_processing/loader.py`**

  * **用途**: 负责从各种来源加载数据集。
  * **包含内容**:

    * `class RealDatasetLoader`

      * `load_opus_datasets()`
      * `load_wmt_datasets()`
      * `load_multi30k_dataset()`
      * `get_fallback()`
      * `load_all_datasets()`

* **`src/pipeline/data_generation_pipeline.py`**

  * **用途**: 组织和驱动整个数据生成流程，是原代码中 `DataGenerationPipeline` 类的归宿。
  * **包含内容**:

    * `class DataGenerationPipeline`
      * `__init__()`
      * `generate_sample_data()`
      * `_calculate_beam_diversity()`
      * `save_data()`

* **`src/analysis/visualization.py`**

  * **用途**: 包含所有用于生成图表的函数。
  * **包含内容**:

    * `create_visualizations(df)`
    * `save_graphical_data(df, filename)`
    * `create_final_visualizations(df, output_dir)`

* **`src/analysis/scientific_analysis.py`**

  * **用途**: 包含更高级的科学分析函数，如相干性视界检测、聚类等。
  * **包含内容**:

    * `class CoherenceHorizonDetector`
    * `analyze_success_vs_failure_patterns(df, strategy, output_dir)`
    * `perform_semantic_clustering_analysis(...)`
    * `generate_comparison_analysis(results, output_dir)`

#### 3. `scripts/` - 可执行脚本

这些脚本是您研究工作的入口点，它们从 `src/` 目录导入模块并执行特定任务。

* **`scripts/01_generate_training_data.py`**

  * **用途**: 专门用于生成训练动力学模型所需的数据集，对应原代码中的 `running_mode = 2`。
  * **逻辑**:
    1. 导入 `DataGenerationPipeline` 和 `config`。
    2. 定义要测试的策略列表。
    3. 调用 `generate_training_data_only` 的逻辑（可以把这个函数移到 `pipeline` 模块中）。
    4. 保存最终的 `strategy_comparison_stepwise.csv` 文件到 `data/processed/`。

* **`scripts/02_run_scientific_experiments.py`**

  * **用途**: 运行完整的科学对比实验，生成所有分析图表和报告，对应原代码中的 `running_mode = 1`。
  * **逻辑**:

    1. 导入 `run_scientific_experiment` 和 `execute_enhanced_analysis` 等分析函数。
    2. 定义要对比的策略。
    3. 执行实验。
    4. 将所有结果（CSV, PNG, JSON）保存到 `results/` 目录。

#### 4. `README.md`

我建议您创建一个 `README.md` 文件，简要介绍项目：

* **项目标题**: A Dynamical Systems Approach to Modeling Exposure Bias Amplification...
* **核心目标**: 引用您 `日程安排.md` 中的核心目标。
* **项目结构**: 简要说明 `src`, `data`, `scripts` 等文件夹的用途。
* **如何运行**:

    1. `pip install -r requirements.txt`
    2. 运行 `scripts/01_generate_training_data.py` 来生成数据。
    3. 运行 `scripts/02_run_scientific_experiments.py` 来复现实验分析。

这个结构将您的研究从一个单一的脚本转变为一个有组织、可维护的科研项目，完全符合您接下来的研究计划。

#-----------------


conda activate E:\computer_science\research\knnmt\paec\venv
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r E:\computer_science\research\knnmt\paec\requirements.txt

pip list --format=freeze | findstr "torch transformers faiss uvicorn"  # Windows
conda list  # 查看所有包（含pip安装的）
python -c "import torch; print(torch.__version__)"  # 测试PyTorch

