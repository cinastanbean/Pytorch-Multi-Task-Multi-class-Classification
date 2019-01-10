
DATA=../data
MAX_BASE_NUMBER=5000

ARC=Yolov2_768x512
CLASS_NUM=24

EPOCHS=120
FC_EPOCHS=50

BATCHSIZE=128
WORKERS=8

LEARNING_RATE=0.01
WEIGHT_DECAY=0.0001

TRAIN_LOG_FILENAME=$ARC"_train_`date +%Y%m%d_%H%M%S`".log
VAL_LOG_FILENAME=$ARC"_val_`date +%Y%m%d_%H%M%S`".log

python main_mtmc_resnet.py --data $DATA \
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
    --arc $ARC \
    --workers $WORKERS \
    --evaluate \
    --resume model_best_checkpoint_$ARC.pth.tar \
    --batch_size $BATCHSIZE \
    2>&1 | tee $VAL_LOG_FILENAME

echo "Val... Done."

