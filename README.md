# LDM-AF: Adaptive Filtering Framework with Learnable Domain Memory for Generalizable Multi-domain Engagement Estimation

This is the official implementation of **LDM-AF**, a framework that achieves state-of-the-art (SOTA) results on the **NoXi-Base**, **NoXi-Add**, and **PinSoRo-CR-Task** datasets within the **MultiMediate'26** challenge. It ranks second in terms of average weighted overall performance.

---

## Environment Setup

Set up the environment using **Python 3.10** and install the required dependencies:
bash
pip install -r requirements.txt

---

## Dataset Directories

Depending on the dataset, navigate to the corresponding directory:

| Directory        | Applicable Datasets                                                      |
|-----------------|--------------------------------------------------------------------------|
| `MM26`          | NoXi-Base, NoXi-Add, NoXi-J                                              |
| `MM26_MPIIGI`   | MPIIGI                                                                    |
| `MM26_PinSoRo`  | PinSoRo-CC-Social, PinSoRo-CC-Task, PinSoRo-CR-Social, PinSoRo-CR-Task    |

---

## Training & Validation

Run `run.py` with the following command:
bash
python -u ./run.py \
--root_path ./LDM-AF \
--gen_data_config_path ./config/gen_data_config.yaml \
--train_config_path ./config/train_config.yaml \
--model_config_path ./config/model_config.yaml \
--validate_config_path ./config/validate_config.yaml

### Outputs (under `--root_path`, i.e. `./LDM-AF`)

| Type              | Format / Content                                                  |
|-------------------|-------------------------------------------------------------------|
| Pre-trained Weights | `.pth` files                                                    |
| Results           | `.npz` files storing final inference outputs & ground-truth labels for the validation set |
| Logs              | `.txt` files documenting training progress                        |
| Visualization     | TensorBoard event files for analyzing training dynamics            |

---

## Analysis & Test Set Prediction

1. List the **absolute paths** to all experiment results under `./LDM-AF` into `analysis.txt`.

2. Run the prediction script to produce final result files for the **MultiMediate'26 test set**:
bash
python -u predict_per_session.py
