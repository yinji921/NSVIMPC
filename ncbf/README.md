Commands to run for collecting offline dataset + training:

# Step 1 - Collect data
In the `robot_planning/scripts` folder:
```bash
python collect_offline_dset.py
```
This collects trajectories and saves the dataset to `data/raw_data.pkl`.

# Step 2 - Process data
In the root folder (`Autorally_MPPI_CLBF`), run:
```bash
python robot_planning/scripts/raw_to_dset.py data/raw_data.pkl
```
This converts the `data/raw_data.pkl` to `data/dset.pkl`.

# Step 3 - Train (offline)
In the root folder (`Autorally_MPPI_CLBF`), run:
```bash
python ncbf/scripts/train_offline.py
```
(Optionally give the run a name, i.e.,)
```bash
python ncbf/scripts/train_offline.py --wandb-name my_name_here
```
This trains the model and saves the checkpoints in the `runs` folder, e.g.,
`runs/offline/0039-disc0.9_fixobo/ckpts/00099999/default`

# Step 4 - Run using the offline trained NCBF.
In the `ncbf.py` file, change the `ckpt_path` variable to the `default` folder of the desired checkpoint, e.g.,
```python
ckpt_path = "/home/oswinso/research/lab/Autorally_MPPI_CLBF/runs/offline/0039-disc0.9_fixobo/ckpts/00099999/default"
```

# Step 5 - Online Training NCBF
Try robot_planning/scripts/run_ar_cbf_online.py
