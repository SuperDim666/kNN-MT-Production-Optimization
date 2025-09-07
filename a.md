# 批次A (Waffle 全责)

```bash
Z1：  --tag "S5Z1" --save_dir "./runs/gnn_S5/S5Z1" --gnn_layer_type gat --gnn_attention_heads 4
Z2：  --tag "S5Z2" --save_dir "./runs/gnn_S5/S5Z2" --gnn_layer_type gcn

# A1｜Softmin 退火（0.5→0.05 / 12ep）— 强退火，倾向更“硬”的选择
python t_train_GNN_S5_Improved.py --tag "S5A15" --save_dir "./runs/gnn_S5/S5A1" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.5 --softmin_tau_final 0.05 --softmin_anneal_ep 12 --rollout_H 12

# A2｜Softmin 退火（0.5→0.12 / 12ep）— 中等退火，控制端点违规同时抑制覆盖下滑
python ./t_train_GNN_S5_Improved.py --tag "S5A2" --save_dir "./runs/gnn_S5/S5A2" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.5 --softmin_tau_final 0.12 --softmin_anneal_ep 12 --rollout_H 12

# A3｜Softmin 退火（0.25→0.05 / 12ep）— 低起点退火，观察早期端点违规压降速度
python ./t_train_GNN_S5_Improved.py --tag "S5A3" --save_dir "./runs/gnn_S5/S5A3" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.25 --softmin_tau_final 0.05 --softmin_anneal_ep 12 --rollout_H 12

# A4｜Softmin 常温（0.12 不退火）— 常温上界对照，剔除退火因素
python ./t_train_GNN_S5_Improved.py --tag "S5A4" --save_dir "./runs/gnn_S5/S5A4" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12

# A5｜Gumbel-ST 退火（1.0→0.1 / 12ep）— 近硬选择，期望更低端点违规
python ./t_train_GNN_S5_Improved.py --tag "S5A5" --save_dir "./runs/gnn_S5/S5A5" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector gumbel_st --gumbel_tau_init 1.0 --gumbel_tau_final 0.1 --gumbel_anneal_ep 12 --rollout_H 12

# A6｜Gumbel-ST 退火（1.0→0.3 / 12ep）— 中等温度，观察覆盖保持能力
python ./t_train_GNN_S5_Improved.py --tag "S5A6" --save_dir "./runs/gnn_S5/S5A6" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector gumbel_st --gumbel_tau_init 1.0 --gumbel_tau_final 0.3 --gumbel_anneal_ep 12 --rollout_H 12

# A7｜Hard-Greedy（无温度/退火）— 绝对硬选择，上界检查端点违规的理论底线
python ./t_train_GNN_S5_Improved.py --tag "S5A7" --save_dir "./runs/gnn_S5/S5A7" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector hard_greedy --rollout_H 12

# A8｜Softmin + ε-greedy（ε:0.30→0.01 / 15ep）— 软选择配合探索，关注覆盖与端点违规的协同
python ./t_train_GNN_S5_Improved.py --tag "S5A8" --save_dir "./runs/gnn_S5/S5A8" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.25 --softmin_tau_final 0.05 --softmin_anneal_ep 12 --use_epsilon_greedy --epsilon_init 0.30 --epsilon_final 0.01 --epsilon_decay_ep 15 --rollout_H 12

# A9｜Gumbel-ST + ε-greedy（ε:0.30→0.01 / 15ep）— 硬化选择配探索，检验是否抑制端点违规同时不牺牲覆盖
python ./t_train_GNN_S5_Improved.py --tag "S5A9" --save_dir "./runs/gnn_S5/S5A9" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector gumbel_st --gumbel_tau_init 1.0 --gumbel_tau_final 0.2 --gumbel_anneal_ep 12 --use_epsilon_greedy --epsilon_init 0.30 --epsilon_final 0.01 --epsilon_decay_ep 15 --rollout_H 12

# A10｜Softmin + 策略熵正则（0.01）— 以分布多样性替代显式 ε-greedy
python ./t_train_GNN_S5_Improved.py --tag "S5A10" --save_dir "./runs/gnn_S5/S5A10" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.25 --softmin_tau_final 0.05 --softmin_anneal_ep 12 --policy_entropy_weight 0.01 --rollout_H 12
```

# 批次B (Waffle 全责)

```bash
# B0｜基线（无调度，仅作锚点） -- 与A4完全相同，请参考A4输出内容

# B1｜中等双调度（ρ:0.2→0.35@8ep；λ:2.0→3.0@10ep）
python ./t_train_GNN_S5_Improved.py --tag "S5B1" --save_dir "./runs/gnn_S5/S5B1" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --rho_final 0.35 --rho_warmup_ep 8 --lambda_clf 2.0 --lambda_clf_final 3.0 --lambda_warmup_ep 10 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12

# B2｜强双调度（ρ:0.2→0.45@8ep；λ:2.0→4.0@10ep）
python ./t_train_GNN_S5_Improved.py --tag "S5B2" --save_dir "./runs/gnn_S5/S5B2" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --rho_final 0.45 --rho_warmup_ep 8 --lambda_clf 2.0 --lambda_clf_final 4.0 --lambda_warmup_ep 10 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12

# B3｜温和长斜坡（ρ:0.2→0.30@12ep；λ:2.0→2.5@12ep）
python ./t_train_GNN_S5_Improved.py --tag "S5B3" --save_dir "./runs/gnn_S5/S5B3" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --rho_final 0.30 --rho_warmup_ep 12 --lambda_clf 2.0 --lambda_clf_final 2.5 --lambda_warmup_ep 12 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12

# B4｜仅 ρ 调度（ρ:0.2→0.40@8ep；λ 固定 2.0）
python ./t_train_GNN_S5_Improved.py --tag "S5B4" --save_dir "./runs/gnn_S5/S5B4" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --rho_final 0.40 --rho_warmup_ep 8 --lambda_clf 2.0 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12

# B5｜仅 λ 调度（λ:2.0→3.5@8ep；ρ 固定 0.2）
python ./t_train_GNN_S5_Improved.py --tag "S5B5" --save_dir "./runs/gnn_S5/S5B5" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --lambda_clf_final 3.5 --lambda_warmup_ep 8 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12

# B6｜延迟双调度（晚期升压：ρ@15ep；λ@16ep）
python ./t_train_GNN_S5_Improved.py --tag "S5B6" --save_dir "./runs/gnn_S5/S5B6" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --rho_final 0.35 --rho_warmup_ep 15 --lambda_clf 2.0 --lambda_clf_final 3.0 --lambda_warmup_ep 16 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12

# B7｜双调度 + 低教师权重（α=0.3）
python ./t_train_GNN_S5_Improved.py --tag "S5B7" --save_dir "./runs/gnn_S5/S5B7" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --teacher_reweight_alpha 0.3 --rho 0.2 --rho_final 0.35 --rho_warmup_ep 8 --lambda_clf 2.0 --lambda_clf_final 3.0 --lambda_warmup_ep 10 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12

# B8｜双调度 + 高教师权重（α=0.7）
python ./t_train_GNN_S5_Improved.py --tag "S5B8" --save_dir "./runs/gnn_S5/S5B8" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --teacher_reweight_alpha 0.7 --rho 0.2 --rho_final 0.35 --rho_warmup_ep 8 --lambda_clf 2.0 --lambda_clf_final 3.0 --lambda_warmup_ep 10 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12

# B9｜双调度 + N-step 权重上调（nstep_lambda=1.5）
python ./t_train_GNN_S5_Improved.py --tag "S5B9" --save_dir "./runs/gnn_S5/S5B9" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --rho_final 0.35 --rho_warmup_ep 8 --lambda_clf 2.0 --lambda_clf_final 3.0 --lambda_warmup_ep 10 --nstep_lambda 1.5 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12

# B10｜双调度 + N-step 权重下调（nstep_lambda=0.5）
python ./t_train_GNN_S5_Improved.py --tag "S5B10" --save_dir "./runs/gnn_S5/S5B10" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --rho_final 0.35 --rho_warmup_ep 8 --lambda_clf 2.0 --lambda_clf_final 3.0 --lambda_warmup_ep 10 --nstep_lambda 0.5 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12
```

# 批次 I｜协同效应（CBF/ADT/Cox 与 Lipschitz）

```bash
# I1｜CBF + ADT（λ_cbf=0.5, α=0.5; λ_adt=0.05）
python ./t_train_GNN_S5_Improved.py --tag "S5I1" --save_dir "./runs/gnn_S5/S5I1" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12 --phi_crit 0.10 --lambda_cbf 0.5 --cbf_alpha 0.5 --lambda_adt 0.05

# I2｜CBF + Cox（λ_cbf=0.5, α=0.5; λ_cox=0.5, thr=1.5）
python ./t_train_GNN_S5_Improved.py --tag "S5I2" --save_dir "./runs/gnn_S5/S5I2" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12 --phi_crit 0.10 --lambda_cbf 0.5 --cbf_alpha 0.5 --lambda_cox 0.5 --cox_event_threshold 1.5

# I3｜CBF + ADT + Cox（0.5/0.05/0.2, thr=2.0）
python ./t_train_GNN_S5_Improved.py --tag "S5I3" --save_dir "./runs/gnn_S5/S5I3" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 1e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12 --phi_crit 0.10 --lambda_cbf 0.5 --cbf_alpha 0.5 --lambda_adt 0.05 --lambda_cox 0.2 --cox_event_threshold 2.0
 
# I4｜协同 + Lipschitz（I3 基础 + 谱归一化 + 强雅可比 5e-3）
python ./t_train_GNN_S5_Improved.py --tag "S5I4" --save_dir "./runs/gnn_S5/S5I4" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 5e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12 --phi_crit 0.10 --lambda_cbf 0.5 --cbf_alpha 0.5 --lambda_adt 0.05 --lambda_cox 0.2 --cox_event_threshold 2.0

# I5｜ADT + Lipschitz（无 CBF/Cox）
python ./t_train_GNN_S5_Improved.py --tag "S5I5" --save_dir "./runs/gnn_S5/S5I5" --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --gnn_layer_type gat --gnn_attention_heads 4 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --rho 0.2 --lambda_clf 2.0 --jacobian_reg 5e-3 --use_nstep_clf --nstep_H 5 --nstep_gamma 0.98 --nstep_selector softmin --softmin_tau 0.12 --rollout_H 12 --phi_crit 0.10 --lambda_adt 0.05
```

## I 组不加：除命令显式开启项外，其余正则/探索/动作扩展不加。
