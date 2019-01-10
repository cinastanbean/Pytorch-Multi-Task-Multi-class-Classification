
DATA=../data
MAX_BASE_NUMBER=5000

ARC=resnet18
CLASS_NUM=24 # deprecated in mtmc
:'
# Restnet:224X224--S7X7--MP7X7--512
DATALOADER_RESIZE_H=256
DATALOADER_RESIZE_W=256
INPUTLAYER_H=224
INPUTLAYER_W=224
FC_FEATURES=512
'
:'
# Inception:320, 299
DATALOADER_RESIZE_H=320
DATALOADER_RESIZE_W=320
INPUTLAYER_H=299
INPUTLAYER_W=299
FC_FEATURES=*
'
# 336X224--S11X7--MP7X7--512*(11-7+1)=512*5=2560
# 960:640 = 3:2 = 224*1.5:224 = 336:224 = 384:256 = 1.5:1
DATALOADER_RESIZE_H=384
DATALOADER_RESIZE_W=256
INPUTLAYER_H=336
INPUTLAYER_W=224
FC_FEATURES=2560

EPOCHS=120
FC_EPOCHS=50

BATCHSIZE=256
WORKERS=8

LEARNING_RATE=0.01
WEIGHT_DECAY=0.0001

TRAIN_LOG_FILENAME=$ARC"_train_`date +%Y%m%d_%H%M%S`".log
VAL_LOG_FILENAME=$ARC"_val_`date +%Y%m%d_%H%M%S`".log

python main_mtmc_resnet.py --data $DATA \
    --dataloader_resize_h $DATALOADER_RESIZE_H \
    --dataloader_resize_w $DATALOADER_RESIZE_W \
    --inputlayer_h $INPUTLAYER_H \
    --inputlayer_w $INPUTLAYER_W \
    --fc_features $FC_FEATURES \
    --max_base_number $MAX_BASE_NUMBER \
    --arc $ARC \
    --workers $WORKERS \
    --pretrained \
    --epochs $EPOCHS \
    --fc_epochs $FC_EPOCHS \
    --batch_size $BATCHSIZE \
    --learning-rate $LEARNING_RATE \
    --weight-decay $WEIGHT_DECAY \
    2>&1 | tee $TRAIN_LOG_FILENAME

echo "Train... Done."

python main_mtmc_resnet.py --data $DATA \
    --dataloader_resize_h $DATALOADER_RESIZE_H \
    --dataloader_resize_w $DATALOADER_RESIZE_W \
    --inputlayer_h $INPUTLAYER_H \
    --inputlayer_w $INPUTLAYER_W \
    --fc_features $FC_FEATURES \
    --arc $ARC \
    --workers $WORKERS \
    --evaluate \
    --resume model_best_checkpoint_$ARC.pth.tar \
    --batch_size $BATCHSIZE \
    2>&1 | tee $VAL_LOG_FILENAME

echo "Val... Done."

