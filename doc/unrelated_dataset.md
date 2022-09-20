# Unrelated Dataset DI
```bash
export CUDA_VISIBLE_DEVICES=2,3

echo $CUDA_VISIBLE_DEVICES

echo "SVHN"
python3 ./src/train.py \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 100 \
  --dataset SVHN \
  --dropRate 0.3

echo "MNIST"
python3 ./src/train.py \
  --batch_size 500 \
  --mode teacher \
  --normalize 1 \
  --model_id teacher_normalized \
  --lr_mode 2 \
  --epochs 100 \
  --dataset MNIST \
  --dropRate 0.3
```