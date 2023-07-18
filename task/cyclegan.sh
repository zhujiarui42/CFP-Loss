/home/servicing/projects/environments/anaconda3/envs/py37_torch_jiarui/bin/python -u main.py \
--model "cyclegan" \
--loss_fn "cycleganloss" \
--batch_size 8 \
--cuda '3' \
--epochs 100 \
--step_size 10 \
--lr 2.e-4 \
--optim "Adam" \
--version "cyclegan"