#!/bin/zsh
# One-shot retrain on the use-everything split (77,729 wallets).
# Runs GCN -> GNN -> GNN eval -> 3-model comparison, sequentially.
set -e
cd "/Users/orimood/Desktop/homework/Visual Studio Code/final_project"
source venv/bin/activate
export PYTHONUNBUFFERED=1
LOG=logs/retrain_useall.log

echo "==== [$(date)] START retrain on use-everything split ====" | tee $LOG

echo "==== [$(date)] 1/4 GCN baseline ====" | tee -a $LOG
python scripts/run_gcn_baseline.py >> $LOG 2>&1
echo "==== [$(date)] GCN done ====" | tee -a $LOG

echo "==== [$(date)] 2/4 GNN training (100 epochs) ====" | tee -a $LOG
python run_pipeline.py --train --epochs 100 --batch-size 64 >> $LOG 2>&1
echo "==== [$(date)] GNN training done ====" | tee -a $LOG

echo "==== [$(date)] 3/4 GNN evaluation ====" | tee -a $LOG
python run_pipeline.py --evaluate >> $LOG 2>&1
echo "==== [$(date)] GNN eval done ====" | tee -a $LOG

echo "==== [$(date)] 4/4 three-model comparison ====" | tee -a $LOG
python scripts/compare_three_models.py >> $LOG 2>&1
echo "==== [$(date)] ALL DONE ====" | tee -a $LOG
