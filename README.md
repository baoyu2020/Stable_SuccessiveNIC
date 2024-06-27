
# Pytorch Implementation of "Stable Successive Neural Image Compression via Coherent Demodulation-based Transformation"

This repository contains the Pytorch implementation of the paper "Stable Successive Neural Image Compression via Coherent Demodulation-based Transformation," submitted to the Signal Processing journal.

## TODO
- [ ] Provide pretrained models
- [ ] Offer supplementary materials

## Architectures
![Architecture Diagram](assets/architecture.png)

## Dependencies
To run this code, you will need the following dependencies:
- Pytorch 1.9
- CompressAI

You can install the necessary dependencies using the following command:
```bash
pip install torch==2.0.1 compressai=1.2.4
```

## Pretrained Models
Pretrained models will be made available soon. Stay tuned for updates.

## Training
To train the models, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/baoyu2020/Stable_SuccessiveNIC.git
    cd Stable_SuccessiveNIC
    cd CDT
    ```
2. Prepare your dataset and adjust the configuration in `config.py`.
3. Run the training script:
    ```bash
    python main_train.py -q 1 --out_dir ./Log/out_dir/ --nt GDN --model M1 --SIC 50 --batch_size 16 --metric mse --lr 1e-4 --epochs 100 --dataset Train_dataset_dir 
    ```

## Testing
To test the models, follow these steps:
1. Run the testing script:
    ```bash
    python main_multi_infere.py -q 1 --nt GDN --model M1 --dataset Test_dataset_dir --save --model_dir  path of trained models --out_dir Log/out_dir/  --SIC 50 
    ```

2. The results will be saved in the `./Log/out_dir/` directory.

## Contact
For any inquiries or issues, please contact us at [ynbao@stu.hit.cn](mailto:ynbao@stu.hit.cn).

---

We hope you find this repository useful. Contributions are welcome!

