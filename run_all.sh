#!/bin/bash

RATIO=${1:-1.0}
EPOCHS=${2:-50}

echo "Running all tasks with ratio=$RATIO and epochs=$EPOCHS"
echo "=============================================="

echo -e "\n[1/5] Running Task 1: Supervised Learning..."
python task1.py $RATIO $EPOCHS

echo -e "\n[2/5] Running Task 2: Contrastive Learning..."
python task2.py $RATIO $EPOCHS

echo -e "\n[3/5] Running Task 3: t-SNE Visualization..."
python task3.py $RATIO

echo -e "\n[4/5] Running Task 4: Voice Biometrics..."
python task4.py $RATIO

echo -e "\n[5/5] Running Task 5: Fine-tuning..."
python task5.py $RATIO $EPOCHS

echo -e "\n=============================================="
echo "All tasks completed!"
echo "Results saved in task{1-5}_* directories"
