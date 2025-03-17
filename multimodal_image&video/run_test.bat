@echo off
set EXPID=best_model
set HOST=127.0.0.1
set PORT=1
set NUM_GPU=1

python test.py ^
--config "configs/test.yaml" ^
--output_dir "results" ^
--launcher pytorch ^
--rank 0 ^
--log_num %EXPID% ^
--dist-url tcp://%HOST%:1003%PORT% ^
--ckpt_path "./checkpoint_best.pth" ^
--token_momentum ^
--world_size %NUM_GPU% ^
--test_epoch best

pause 