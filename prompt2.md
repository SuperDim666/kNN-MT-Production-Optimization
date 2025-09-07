
python ./t_train_GNN_S5_Improved.py --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --jacobian_reg 1e-3 --use_nstep_clf --action_search perturb --num_delta_dirs 2 --action_delta 0.10

python ./t_train_GNN_S5_Improved.py --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --jacobian_reg 1e-3 --use_nstep_clf 

----

python ./t_train_GNN_S5_Improved.py --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --jacobian_reg 1e-3 --use_nstep_clf --action_search perturb --num_delta_dirs 2 --action_delta 0.10 --gnn_layer_type gcn --tag "S6A1" --save_dir "./runs/gnn_S6/S6A1"

python ./t_train_GNN_S5_Improved.py --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --jacobian_reg 1e-3 --use_nstep_clf --action_search perturb --num_delta_dirs 2 --action_delta 0.10 --gnn_layer_type gat --gnn_attention_heads 4 --tag "S6A2" --save_dir "./runs/gnn_S6/S6A2"

python ./t_train_GNN_S5_Improved.py --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --jacobian_reg 1e-3 --use_nstep_clf --action_search perturb --num_delta_dirs 2 --action_delta 0.10 --gnn_layer_type rgcn --gnn_num_relations 5 --tag "S6A3" --save_dir "./runs/gnn_S6/S6A3"

---


# 比较
 --nstep_aggr endpoint --tag "S6B1" --save_dir "./runs/gnn_S6/S6B1"

 --nstep_aggr endpoint --nstep_lambda 1.5 --tag "S6B2" --save_dir "./runs/gnn_S6/S6B2"

 --nstep_H 9 --nstep_bptt_window 5 --rollout_H 32 --tag "S6B3" --save_dir "./runs/gnn_S6/S6B3"


# B2 -- 与GAT与A3
python ./t_train_GNN_S5_Improved.py --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --jacobian_reg 1e-3 --use_nstep_clf --action_search perturb --num_delta_dirs 2 --action_delta 0.10 --gnn_layer_type gat --gnn_attention_heads 4 --tag "S6B2" --save_dir "./runs/gnn_S6/S6B2"

python ./t_train_GNN_S5_Improved.py --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --jacobian_reg 1e-3 --use_nstep_clf --action_search perturb --num_delta_dirs 2 --action_delta 0.10 --gnn_layer_type gat --gnn_attention_heads 4 --tag "S6B2" --save_dir "./runs/gnn_S6/S6B2"

# 最终预估 - C
## C0 开启lipschitz, GAT模型，BOTH动作选择，ENDPOINT
python ./t_train_GNN_S5_Improved.py --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --jacobian_reg 1e-3 --use_nstep_clf --action_search both --num_delta_dirs 2 --action_delta 0.10 --gnn_layer_type gat --gnn_attention_heads 4 --nstep_aggr endpoint --tag "S6C0" --save_dir "./runs/gnn_S6/S6C0"

# C1 | nstep_H=3 (近距)
python ./t_train_GNN_S5_Improved.py --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --jacobian_reg 1e-3 --use_nstep_clf --action_search both --num_delta_dirs 2 --action_delta 0.10 --gnn_layer_type gat --gnn_attention_heads 4 --nstep_aggr endpoint --nstep_H 3 --tag "S6C1" --save_dir "./runs/gnn_S6/S6C1"

# C2 | nstep_H=6 (近距)
python ./t_train_GNN_S5_Improved.py --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --jacobian_reg 1e-3 --use_nstep_clf --action_search both --num_delta_dirs 2 --action_delta 0.10 --gnn_layer_type gat --gnn_attention_heads 4 --nstep_aggr endpoint --nstep_H 6 --tag "S6C2" --save_dir "./runs/gnn_S6/S6C2"

# C3 | nstep_H=9 (远距)
python ./t_train_GNN_S5_Improved.py --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --jacobian_reg 1e-3 --use_nstep_clf --action_search both --num_delta_dirs 2 --action_delta 0.10 --gnn_layer_type gat --gnn_attention_heads 4 --nstep_aggr endpoint --nstep_H 9 --tag "S6C3" --save_dir "./runs/gnn_S6/S6C3"

# C4 | nstep_H=12 (超远距)
python ./t_train_GNN_S5_Improved.py --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --jacobian_reg 1e-3 --use_nstep_clf --action_search both --num_delta_dirs 2 --action_delta 0.10 --gnn_layer_type gat --gnn_attention_heads 4 --nstep_aggr endpoint --nstep_H 9 --tag "S6C4" --save_dir "./runs/gnn_S6/S6C4"

## C5 测试NCLF - 弱nstep_lambda
python ./t_train_GNN_S5_Improved.py --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --jacobian_reg 1e-3 --use_nstep_clf --action_search both --num_delta_dirs 2 --action_delta 0.10 --gnn_layer_type gat --gnn_attention_heads 4 --nstep_aggr endpoint --nstep_lambda 0.8 --tag "S6C1" --save_dir "./runs/gnn_S6/S6C1"

## C6 测试NCLF - 强nstep_lambda
python ./t_train_GNN_S5_Improved.py --train_path ./strategy_comparison_stepwise.csv --val_ratio 0.2 --format csv --batch_size 512 --adj_mode tri --learn_P_diag --P_init 1.0 1.0 1.0 --export_action_stats --export_rollout_csv --earlystop_mode stability_first --stab_lexi_eps 1e-3 --use_spectral_norm --epochs 20 --lr 1e-3 --weight_decay 1e-4 --jacobian_reg 1e-3 --use_nstep_clf --action_search both --num_delta_dirs 2 --action_delta 0.10 --gnn_layer_type gat --gnn_attention_heads 4 --nstep_aggr endpoint --nstep_lambda 1.5 --tag "S6C2" --save_dir "./runs/gnn_S6/S6C2"