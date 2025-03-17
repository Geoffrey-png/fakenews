$EXPID = "best_model"
$MYHOST = "127.0.0.1"
$PORT = "1"
$NUM_GPU = 1

python test.py `
--config "configs/test.yaml" `
--output_dir "results" `
--launcher pytorch `
--rank 0 `
--log_num $EXPID `
--dist-url "tcp://${MYHOST}:1003${PORT}" `
--dist-backend "gloo" `
--checkpoint "./checkpoint_best.pth" `
--token_momentum `
--world_size $NUM_GPU `
--test_epoch best 