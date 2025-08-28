# **S5 运行报告**

## **基础指令**

```bash
BASE=$(cat <<EOF
t_train_GNN_S5.py \
--train_path ./strategy_comparison_stepwise.csv \
--format csv \
--epochs 20 \
--batch_size 512 \
--val_ratio 0.2 \
--adj_mode tri \
--hid_dim 64 \
--layers 3 \
--gnn_layer_type gat --gnn_attention_heads 4 \
--learn_P_diag --P_init 1.0 1.0 1.0 \
--action_search both --num_delta_dirs 2 --action_delta 0.25 \
--earlystop_mode stability_first --stab_lexi_eps 1e-3 \
--use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_bptt_window -1 \
--nstep_selector gumbel_st --gumbel_tau_init 1.0 --gumbel_tau_final 0.1 --gumbel_anneal_ep 15 \
--use_cvar_loss --cvar_alpha 0.8 \
--policy_entropy_weight 0.01 \
--rollout_H 20 \
--jacobian_reg 1e-3 \
--teacher_reweight_alpha 0.5 \
--rho 0.2 \
--export_action_stats --export_rollout_csv
EOF
)
```

## **批次A：隔离验证 $\mathcal{L}_{\text{\textbf{CBF}}}$ (控制障碍函数)**

**目的：** 验证 $\text{CBF}$ 软约束的效果，并测试不同权重的影响。

**观察指标：** `cbf_loss_teacher_val`, `clf_violation_rate`, 各大 `rmse` 与 `w1`。

### **批次A1：弱 $\text{\textbf{CBF}}$ 约束**

#### **指令（S5-批次A1）**

```bash
python $BASE --lambda_cbf 1.0 --phi_crit 1.5 --cbf_alpha 0.3 --tag "S5A1_CBF_weak" --save_dir "./runs/gnn_S5/S5A1_CBF_weak"
```

### **批次A2：中等 $\text{\textbf{CBF}}$ 约束**

#### **指令（S5-批次A2）**

```bash
python $BASE --lambda_cbf 5.0 --phi_crit 1.5 --cbf_alpha 0.3 --tag "S5A2_CBF_medium" --save_dir "./runs/gnn_S5/S5A2_CBF_medium"
```

### **批次A3：强 $\text{\textbf{CBF}}$ 约束**

#### **指令（S5-批次A3）**

```bash
python $BASE --lambda_cbf 20.0 --phi_crit 1.5 --cbf_alpha 0.3 --tag "S5A3_CBF_strong" --save_dir "./runs/gnn_S5/S5A3_CBF_strong"
```

### **批次A4：不同安全边界 $\text{\textbf{CBF}}$ 约束**

#### **指令（S5-批次A4）**

```bash
python $BASE --lambda_cbf 5.0 --phi_crit 1.2 --cbf_alpha 0.3 --tag "S5A4_CBF_strict_boundary" --save_dir "./runs/gnn_S5/S5A4_CBF_strict_boundary"
```

## **批次B：隔离验证 $\mathcal{L}_{\text{\textbf{ADT}}}$ (平均停留时间)**

**目的：** 验证 $\text{ADT}$ 惩罚对策略平滑性的影响。

**观察指标：** `adt_switch_rate` (最关键), `clf_violation_rate` (次要影响)。

### **批次B1：弱 $\text{\textbf{ADT}}$ 惩罚**

#### **指令（S5-批次B1）**

```bash
python $BASE --lambda_adt 0.05 --tag "S5B1_ADT_weak" --save_dir "./runs/gnn_S5/S5B1_ADT_weak"
```

### **批次B2：中等 $\text{\textbf{ADT}}$ 惩罚**

#### **指令（S5-批次B2）**

```bash
python $BASE --lambda_adt 0.2 --tag "S5B2_ADT_medium" --save_dir "./runs/gnn_S5/S5B2_ADT_medium"
```

### **批次B3：强 $\text{\textbf{ADT}}$ 惩罚**

#### **指令（S5-批次B3）**

```bash
python $BASE --lambda_adt 1.0 --tag "S5B3_ADT_strong" --save_dir "./runs/gnn_S5/S5B3_ADT_strong"
```

## **批次C：隔离验证 $\mathcal{L}_{\text{\textbf{cox}}}$ ($\text{\textbf{Cox}}$ 风险模型)**

**目的：** 验证 $\text{Cox}$ 损失能否引导模型规避高风险状态。

**观察指标：** `cox_loss_val`, `nstep_endpoint_violation_rate` (预期会下降)。

### **批次C1：弱 $\text{\textbf{Cox}}$ 惩罚**

#### **指令（S5-批次C1）**

```bash
python $BASE --lambda_cox 0.5 --tag "S5C1_Cox_weak" --save_dir "./runs/gnn_S5/S5C1_Cox_weak"
```

### **批次C2：中等 $\text{\textbf{Cox}}$ 惩罚**

#### **指令（S5-批次C2）**

```bash
python $BASE --lambda_cox 2.0 --tag "S5C2_Cox_medium" --save_dir "./runs/gnn_S5/S5C2_Cox_medium"
```

### **批次C3：强 $\text{\textbf{Cox}}$ 惩罚**

#### **指令（S5-批次C3）**

```bash
python $BASE --lambda_cox 10.0 --tag "S5C3_Cox_strong" --save_dir "./runs/gnn_S5/S5C3_Cox_strong"
```

### **批次C4：不同的事件阈值**

#### **指令（S5-批次C4）**

```bash
python $BASE --lambda_cox 2.0 --cox_event_threshold 1.5 --tag "S5C4_Cox_strict_event" --save_dir "./runs/gnn_S5/S5C4_Cox_strict_event"
```

## **批次D：验证 $\text{\textbf{Lipschitz}}$ 正则化**

**目的：** 验证谱归一化对模型泛化和稳定性的影响。

**观察指标：** 对比有/无正则化的验证集 `rmse`、`w1`、`clf_violation_rate` 和 其他指标的平滑度。

### **批次D1：$\text{\textbf{GAT}}$ 基线 (无谱归一化)**

#### **指令（S5-批次D1）**

```bash
python $BASE --tag "S5D1_GAT_no_SN" --save_dir "./runs/gnn_S5/S5D1_GAT_no_SN"
```

### **批次D2：$\text{\textbf{GAT}}$ + 谱归一化**

#### **指令（S5-批次D2）**

```bash
python $BASE --use_spectral_norm --tag "S5D2_GAT_with_SN" --save_dir "./runs/gnn_S5/S5D2_GAT_with_SN"
```

### **批次D3：$\text{\textbf{RGCN}}$ + 谱归一化 (验证对不同架构的通用性)**

#### **指令（S5-批次D3）**

```bash
python $BASE --use_spectral_norm --gnn_layer_type rgcn --gnn_num_relations 5 --tag "S5D3_RGCN_with_SN" --save_dir "./runs/gnn_S5/S5D3_RGCN_with_SN"
```

## **批次E：组合测试与最终模型**

**目的：** 将各项有效的功能组合，寻找最优的“完全体”模型。

**观察指标：** 所有 S5 核心指标的综合表现。

### **批次E1：组合 $\text{\textbf{CBF}}$ + $\text{\textbf{ADT}}$**

#### **指令（S5-批次E1）**

```bash
python $BASE --lambda_cbf 5.0 --phi_crit 1.5 --cbf_alpha 0.3 --lambda_adt 0.2 --tag "S5E1_CBF_ADT" --save_dir "./runs/gnn_S5/S5E1_CBF_ADT"
```

### **批次E2：组合 $\text{\textbf{CBF}}$ + $\text{\textbf{Cox}}$**

#### **指令（S5-批次E2）**

```bash
python $BASE --lambda_cbf 5.0 --phi_crit 1.5 --cbf_alpha 0.3 --lambda_cox 2.0 --tag "S5E2_CBF_Cox" --save_dir "./runs/gnn_S5/S5E2_CBF_Cox"
```

### **批次E3：完全体 S5 模型 ($\text{\textbf{CBF}}$ + $\text{\textbf{ADT}}$ + $\text{\textbf{Cox}}$)**

#### **指令（S5-批次E3）**

```bash
python $BASE --lambda_cbf 5.0 --phi_crit 1.5 --cbf_alpha 0.3 --lambda_adt 0.2 --lambda_cox 2.0 --tag "S5E3_Full_Model" --save_dir "./runs/gnn_S5/S5E3_Full_Model"
```

### **批次E4：完全体 S5 模型 + $\text{\textbf{Lipschitz}}$ 正则化 (最终候选)**

#### **指令（S5-批次E4）**

```bash
python $BASE --lambda_cbf 5.0 --phi_crit 1.5 --cbf_alpha 0.3 --lambda_adt 0.2 --lambda_cox 2.0 --use_spectral_norm --tag "S5E4_Full_Model_SN" --save_dir "./runs/gnn_S5/S5E4_Full_Model_SN"
```
