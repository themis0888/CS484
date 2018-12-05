## Dependencies

- Tensorflow 1.4
- You can install other python packages by running following command
  `pip install -r requirements.txt` 
  (sk image, numpy ...)



## Usage

Put your data in followin format 

- Training

    ```shell
    CUDA_VISIBLE_DEVICES=0 python -i training.py \
    --data_path=/home/siit/navi/data/input_data/60fps/ \
    --im_size=128 --batch_size=4 --ratio=2 \
    --mode=training --checkpoint_path=./01checkpoints \
    --model_mode=cbsr --sample_path=1127cbsr_tr \
    ```

    You can find the log file from `./log`, output sample from `./sample`, and check point from `./checkpoints`

     

- Testing 

    ```shell
    CUDA_VISIBLE_DEVICES=0 python -i training.py \
    --data_path=/home/siit/navi/data/input_data/60fps/ \
    --im_size=128 --batch_size=4 --ratio=2 \
    --mode=testing --checkpoint_path=./01checkpoints \
    --model_mode=cbsr --sample_path=1127cbsr_tr \
    ```

    In order to run this, you need to put the checkpoint file in `./checkpoint`

    You can find the result file from `./log`, output images from `./sample`


