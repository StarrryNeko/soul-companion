# 测试与模型评估证据采集指南

以下命令建议在云端 GPU 环境、项目根目录执行。模型权重不提交，但文本结果、CSV、JSON、JUnit XML 和图表应保留。

## 1. 数据与自动化测试

```bash
mkdir -p artifacts
python scripts/validate_data.py | tee artifacts/data_validation.txt
pytest -q -ra --junitxml=artifacts/pytest.xml | tee artifacts/pytest_output.txt
```

截图要求：终端中同时显示完整命令、`20 passed` 和耗时。建议文件名：`test_pytest.png`。

## 2. 16条端到端场景

先确认需要演示的模型已经加载，或设置好 `DEEPSEEK_API_KEY`，再运行：

```bash
python scripts/evaluate.py \
  --output artifacts/pipeline_case_results.json \
  | tee artifacts/pipeline_case_output.txt
```

截图要求：终端末尾显示 `evaluated=16`、JSON 输出路径，并保留 `pipeline_case_results.json`。建议文件名：`test_pipeline16.png`。

运行后必须检查 JSON 中每条记录的 `generation_backend`。如果大量记录为 `template`，不能将结果当作微调模型效果。

## 3. 参数实验训练

```bash
llamafactory-cli train config/llamafactory_train_r8.yaml
llamafactory-cli train config/llamafactory_train_regularized.yaml
llamafactory-cli train config/llamafactory_train_all_linear.yaml
llamafactory-cli train config/llamafactory_train_qlora.yaml
```

每次训练开始前和完成后保存：

```bash
nvidia-smi | tee artifacts/nvidia_smi_<experiment>.txt
```

另外保存输出目录中的 `trainer_state.json`。建议记录 GPU 型号、训练总时长、峰值显存、最终训练 loss 和最低验证 loss。

## 4. 统一模型评估

```bash
python scripts/evaluate_model_comparison.py \
  --model base=base:../models/Qwen2.5-1.5B-Instruct \
  --model baseline=adapter:output/lora_adapter \
  --model r8=adapter:output/lora_adapter_r8 \
  --model regularized=adapter:output/lora_adapter_regularized \
  --model all_linear=adapter:output/lora_adapter_all_linear \
  --model qlora=adapter:output/lora_adapter_qlora \
  | tee artifacts/model_evaluation_output.txt
```

必须保留：

- `responses.json`
- `case_scores.csv/json`
- `summary_by_model.csv/json`
- `summary_by_category.csv/json`
- `overall_score_comparison.png`
- `category_score_comparison.png`

截图建议：

1. 20题总体均分横向排序图，文件名 `eval_overall.png`。
2. 五个能力维度热力图，文件名 `eval_categories.png`。
3. 至少一组基础模型与最佳 adapter 的原始回答对照，文件名 `eval_response_example.png`。
4. LoRA 与 QLoRA 显存对比证据，文件名 `eval_vram.png`。

## 5. 截图质量检查

- 不裁掉命令、模型名称、坐标轴或图例。
- 不只展示成功案例；至少展示一个失败或边界案例。
- 截图中的数字必须能在对应 CSV/JSON 中找到。
- 运行结果变化后应同步更新报告，禁止继续使用旧图。
