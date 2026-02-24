# Formal Verification for Deep Learning-based Mobile Network Traffic Prediction

### Installation

```bash
cd code/core
pip3 install -r requirements
```

### Usage

1. Train DeepCog model
```bash
python3 main.py
```

2. Export onnx
```
python3 main.py --scenario=export_onnx
```

3. Export vnnlib
```
python3 main.py --scenario=export_vnnlib
```

4. Run verification experiment
```bash
python3 main.py --scenario=prepare_csv
git submodule update --init --recursive
python3 main.py --scenario=verify
```

5. Extract results tables at `../data/csv/verify.csv` or running
```bash
python3 main.py --scenario=table1 # extract under estimation result
python3 main.py --scenario=table2 # extract over estimation result
```
