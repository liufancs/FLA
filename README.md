# FLA

This is our implementation for the paper:

Zhiyong Cheng, Fan Liu, Shenghan Mei, Yangyang Guo, Lei Zhu and Liqiang Nie. Feature-level Attentive ICF for Recommendation. Transactions on Information Systems. 

## Environment Settings
- Tensorflow-gpu version:  1.3.0

## Example to run the codes.
Run FISM.py
```
python FISM.py --dataset delicious --batch_choice user --embed_size 16  --lr 0.01
```

```
python FISM.py --dataset Beauty --batch_choice user --embed_size 16  --lr 0.01
```

# delicious
Run NAIS.py
```
python NAIS.py --dataset delicious --batch_choice user --weight_size 16 --embed_size 16  --pretrain 0 --lr 0.01 --beta 0.7
```

```
python NAIS.py --dataset delicious --batch_choice user --weight_size 16 --embed_size 16  --pretrain 1 --prelr 0.01 --lr 0.01 --beta 0.7
```
Run NAIS1.py
```
python NAIS1.py --dataset delicious --batch_choice user --weight_size 16 --embed_size 16  --pretrain 0 --lr 0.01 --beta 0.7
```

```
python NAIS1.py --dataset delicious --batch_choice user --weight_size 16 --embed_size 16  --pretrain 1 --prelr 0.01 --lr 0.01 --beta 0.7
```

Run NAIS2.py
```
python NAIS2.py --dataset delicious --batch_choice user --weight_size 16 --embed_size 16  --pretrain 0 --lr 0.01 --beta 0.7
```

```
python NAIS2.py --dataset delicious --batch_choice user --weight_size 16 --embed_size 16  --pretrain 1 --prelr 0.01 --lr 0.01 --beta 0.7
```
