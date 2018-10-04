## Dependencies

- Tensorflow 1.4
- You can install other python packages by running following command
  `pip install -r requirements.txt` 
  (sk image, numpy ...)


- Feature extraction

    ```shell
    CUDA_VISIBLE_DEVICES=0 \
    python -i feature_extractor.py \
    --data_path=/home/siit/navi/data/input_data/ukbench_small/ \
    --save_path=/home/siit/navi/data/meta_data/ukbench_small/ \
    --model_name=vgg_16
    ```


     

- Testing 

    ```shell
    CUDA_VISIBLE_DEVICES=0 \
    python -i search.py \
    --data_path=/home/siit/navi/data/input_data/ukbench_small/ \
    --meta_path=/home/siit/navi/data/meta_data/ukbench_small/ \
    --model_name=vgg_16
    ```


