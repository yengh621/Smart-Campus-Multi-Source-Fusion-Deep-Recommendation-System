# MOOCCubeX 与泰迪杯数据对齐脚本

运行：

```powershell
python build_cross_dataset.py
```

本地默认输入为：

- `MOOCCubeX/MOOCCubeX-main/entities/user.json`
- `MOOCCubeX/MOOCCubeX-main/entities/course.json`
- `MOOCCubeX/MOOCCubeX-main/relations/user-problem.json`
- `dd0f7-main/data1.csv`（学生）
- `dd0f7-main/data2.csv`（消费）
- `dd0f7-main/data3.csv`（门禁）

默认输出到 `aligned_output/`。若目录非空，脚本会停止；确认覆盖时使用：

```powershell
python build_cross_dataset.py --overwrite
```

若运行已生成 `all_global_id_mapping.csv`、`sampled_mooc_user.json` 和
`sampled_user_problem.json`，但在后续图谱裁剪中断，可跳过全量 MOOC 扫描继续：

```powershell
python build_cross_dataset.py --output my_output --resume
```

`--resume` 与 `--overwrite` 不能同时使用。

泰迪杯原始 CSV 是 GB18030 编码，脚本会自动识别并统一输出 UTF-8。门禁表通过
`student.AccessCardNo -> door.AccessCardNo` 关联，消费表通过
`student.CardNo -> consume.CardNo` 关联。

## 重要口径

MOOCCubeX 用户没有专业字段。脚本遍历 `course_order`，反查每门课程的 `field`，
统计原始 field 频次并取最高频项，再归并到八大学科。输出的 `subject` 是推导标签，
不是用户真实专业。

两套数据没有真实的用户对应关系。`global_user_id` 是“同学科约束下的模拟配对键”，
论文中不可写成真实身份融合。

默认按任务中的“僵尸学生”口径，仅剔除消费和门禁两者皆无的学生，即二者至少一项
活动就算有效。若希望采用更严格的“两者都必须有”，可运行：

```powershell
python build_cross_dataset.py --teddy-activity both
```

专业映射可用 JSON 覆盖：

```powershell
python build_cross_dataset.py --major-map-override teddy_major_override.example.json
```

## 主要输出

- `all_global_id_mapping.csv`
- `sampled_mooc_user.json`
- `sampled_user_problem.json`
- `mapped_teddy_student.csv`
- `mapped_teddy_consume.csv`
- `mapped_teddy_door.csv`
- `sampled_course.json`、`sampled_problem.json`、`sampled_concept.json`
- `sampled_concept_problem.txt`、`sampled_concept_course.txt`
- `sampled_prerequisites/*.json`
- `subject_dist_stat.csv`
- `teddy_major_subject_mapping.csv`（专业归类审计表）

若缺少可选的关系或先修文件，脚本会跳过对应裁剪，其余输出继续生成；补入标准原始文件后
重新运行即可获得相应文件。损坏的个别 JSONL 行也会静默跳过，只有导致任务中止的问题才会
以 `[失败]` 显示。
