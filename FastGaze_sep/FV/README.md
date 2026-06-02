python train.py --dataset_dir=/zjh/data/FastGaze-cocosearch --img_ftrs_dir=/zjh/data/FastGaze-cocosearch/image_features \
 --train_file coco_search18_fixations_FV_train.json --valid_file coco_search18_fixations_FV_valid.json --model_root ./saved_models/trained-FV --condition freeview \
 --net_name adaptgaze7 --sc_ior True --max_len 7 --num_encoder 3 --head_lr 1e-6 --tail_lr 1e-6 --belly_lr 2e-6 \
 --num_decoder 3 --hidden_dim 512 --lm_hidden_dim 512 --img_hidden_dim 2048 --batch_size 16 --epoch 200 --cuda=7 



python test.py --trained_model=./model_zoo/fastgaze-T.pkg \
--sc_mask True --sc_ior True  --max_len 7 \
--lm_hidden_dim=512 --num_encoder 2 --num_decoder 2 --hidden_dim 256 --img_hidden_dim 2048 \
--dataset_dir=/zjh/data/FastGaze-cocosearch --img_ftrs_dir=/zjh/data/FastGaze-cocosearch/image_features --cuda=6

python test.py --trained_model=./model_zoo/fastgaze-S.pkg \
--sc_mask True --sc_ior True  --max_len 7 \
--lm_hidden_dim=512 --num_encoder 3 --num_decoder 3 --hidden_dim 256 --img_hidden_dim 2048 \
--dataset_dir=/zjh/data/FastGaze-cocosearch --img_ftrs_dir=/zjh/data/FastGaze-cocosearch/image_features --cuda=5

python test.py --trained_model=./model_zoo/fastgaze-B.pkg \
--sc_mask True --sc_ior True  --max_len 7 \
--lm_hidden_dim=512 --num_encoder 3 --num_decoder 3 --hidden_dim 512 --img_hidden_dim 2048 \
--dataset_dir=/zjh/data/FastGaze-cocosearch --img_ftrs_dir=/zjh/data/FastGaze-cocosearch/image_features --cuda=4


python plot_scanpath1.py --trained_model=FastGazeB --condition=freeview \
--dataset_dir=/zjh/data/FastGaze-cocosearch --task microwave --imgfile 000000091615.jpg \
--sc_mask True --sc_ior True --max_len 7 --emlength 7 --lm_hidden_dim 512 --hidden_dim 512 --num_encoder 3 --num_decoder 3 --img_hidden_dim 2048 --cuda=4

