## Dependencies

- Tensorflow 1.4
- You can install other python packages by running following command
  `pip install -r requirements.txt` 
  (sk image, numpy ...)



## Usage

Put your data in followin format 

```bash
/mscoco
├── test
│   ├── HR
│   └── LR
└── train
    ├── HR
    └── LR
```



- Training

    ```shell
    CUDA_VISIBLE_DEVICES=1 python -i training.py \
    --data_path=/home/siit/navi/data/input_data/mscoco/ \
    --im_size=64 --batch_size=16 --ratio=2 \
    --mode=training --checkpoint_path=./checkpoint \
    ```

    You can find the log file from `./log`, output sample from `./sample`, and check point from `./checkpoints`

     

- Testing 

    ```shell
    CUDA_VISIBLE_DEVICES=1 python -i testing.py \
    --data_path=/home/siit/navi/data/input_data/mscoco/ \
    --im_size=64 --batch_size=16 --ratio=2 \
    --mode=testing --checkpoint_path=./checkpoint \
    ```

    In order to run this, you need to put the checkpoint file in `./checkpoint`

    You can find the result file from `./log`, output images from `./sample`


