# SoulCompanion 心晴

面向大学生的心理健康科普与情绪支持助手。项目用于课程演示，包含 LLaMA-Factory 微调配置、RAG 知识库、安全危机识别、情绪记录工具、呼吸练习工具和 Gradio 前端。

核心技术栈：LLaMA-Factory（SFT + LoRA）+ Gradio + ChromaDB。

## 重要声明

本项目仅用于学习和课程展示，不提供心理咨询、医学诊断、治疗或用药建议。若用户出现自伤、自杀或即时危险信号，系统应优先输出现实支持和专业转介信息。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/validate_data.py
python scripts/build_knowledge_base.py
python src/main.py
```

python -m src.main

打开 Gradio 页面：

```text
http://127.0.0.1:7860
```

## LLaMA-Factory 微调

推荐在云端 GPU 环境运行，例如 L4 24GB 或 A10 24GB。

```bash
conda create -n soul python=3.10 -y
conda activate soul
pip install -r requirements.txt

# 训练 LoRA Adapter，输出到 output/lora_adapter/
llamafactory-cli train config/llamafactory_train.yaml

# 训练增加 steps 的 LoRA Adapter，输出到 output/lora_adapter_long_steps/
# 该配置不会覆盖原始 adapter，可用于和短 steps 模型做对比。
llamafactory-cli train config/llamafactory_train_long_steps.yaml

# 训练中等 steps 的 LoRA Adapter，输出到 output/lora_adapter_medium_steps/
# 该配置将 max_steps 调整为 600，用于降低 long steps 过拟合风险并做三模型对比。
llamafactory-cli train config/llamafactory_train_medium_steps.yaml

# 可选：导出合并模型，输出到 output/merged_model/
llamafactory-cli export config/llamafactory_export.yaml

# 可选：导出增加 steps 后的合并模型，输出到 output/merged_model_long_steps/
llamafactory-cli export config/llamafactory_export_long_steps.yaml

# 可选：导出中等 steps 后的合并模型，输出到 output/merged_model_medium_steps/
llamafactory-cli export config/llamafactory_export_medium_steps.yaml

# 可选：基于合并模型导出 INT4，输出到 output/merged_model_int4/
llamafactory-cli export config/llamafactory_export_int4.yaml
```

训练数据采用 LLaMA-Factory Alpaca 格式，注册文件位于 `data/training/dataset_info.json`。

## 微调效果评估对比

新增评估脚本会对基础模型、原短 steps LoRA、中等 steps LoRA、增加 steps LoRA 使用同一组测试题生成回答，并输出明细表、汇总表和图表：

```bash
python scripts/evaluate_model_comparison.py
```

训练完成后，也可以直接绘制 short、medium、long 三个 adapter 的 loss 曲线对比：

```bash
python scripts/plot_training_loss_comparison.py
```

默认输出目录为 `output/evaluation_comparison/`，包含：

- `responses.json`：各模型原始回答
- `case_scores.csv`：逐题评分明细
- `summary_by_model.csv`：模型整体平均分
- `summary_by_category.csv`：不同能力维度平均分
- `case_scores.json`：UTF-8 逐题评分明细
- `summary_by_model.json`：UTF-8 模型整体平均分
- `summary_by_category.json`：UTF-8 不同能力维度平均分
- `overall_score_comparison.png`：整体评分柱状图
- `category_score_comparison.png`：分能力维度柱状图
- `training_loss_comparison.png`：短、中、长 steps 训练 loss 曲线图

如果已提前生成回答，也可以只复用回答文件重新打分和出图：

```bash
python scripts/evaluate_model_comparison.py --responses-json output/evaluation_comparison/responses.json
```

## 目录结构

```text
soul-companion/
  config/
    llamafactory_train.yaml
    llamafactory_train_medium_steps.yaml
    llamafactory_train_long_steps.yaml
    llamafactory_export.yaml
    llamafactory_export_medium_steps.yaml
    llamafactory_export_long_steps.yaml
    llamafactory_export_int4.yaml
  data/
    evaluation/
      model_eval_cases.json
    training/
      dataset_info.json
      mental_health_qa.json
    knowledge_base/
  src/
  scripts/
  tests/
```

## 数据说明

- `data/training/mental_health_qa.json`：Alpaca 格式微调样本。
- `data/training/dataset_info.json`：LLaMA-Factory 数据集注册。
- `data/knowledge_base/docs/`：RAG 文档。
- `config/crisis_keywords.json`：危机关键词和固定回复模板。
- `tests/test_cases.json`：端到端行为测试用例。
