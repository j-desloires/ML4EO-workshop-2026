import os
from pathlib import Path

project_root = Path(__file__).parents[2]  # set to this if running script from repo root
# project_root = Path("../..").resolve() # set to this if running as notebook
print(project_root)

# ---
# S2 fine-tuning (on sample dataset)
config_file = project_root / "config/config-ibm-geospatial-saltmarsh-uk-s2-extent.yaml"
output_dir = project_root / "data/finetune/s2"
command = f"""terratorch fit \
    --config {config_file} \
    --trainer.default_root_dir {output_dir} \
    --trainer.max_epochs 1 \
    --data.init_args.batch_size 2"""

print(f"{command=}")
# os.system(command)

# get hold of pretrained weights
checkpoint_path = (
    project_root
    / "checkpoints/ibm-geospatial-saltmarsh-uk-s2-extent-10m_state_dict.ckpt"
)

# S2 testing
test_images = ...
test_labels = ...
command = f"terratorch test --config {config_file}\
    --ckpt_path {checkpoint_path} \
    --trainer.default_root_dir {output_dir}"
print(f"{command=}")
# os.system(command)

# S2 inference (on full dataset weights)
test_images = project_root / "data/inference/images/s2"
inference_output = project_root / "data/inference/pred/s2"
predict_command = (
    f"terratorch predict \
    --config {config_file} \
    --ckpt_path {checkpoint_path} --trainer.default_root_dir {output_dir} \
    --predict_output_dir {inference_output} \
    --data.init_args.predict_data_root.S2L2A {test_images} "
    ""
)
print(f"{predict_command=}")
# os.system(predict_command)

# S2 inference visualisation
...

## additional modality example ---
# best model inference - 2m RGB + DEM extent
checkpoint_path = (
    project_root
    / "checkpoints/ibm-geospatial-saltmarsh-uk-rgbdem-extent-2m_state_dict.ckpt"
)
config_file = (
    project_root / "config/config-ibm-geospatial-saltmarsh-uk-rgbdem-extent-2m.yaml"
)
output_dir = project_root / "data/inference/pred/rgb_dem_extent/logs"
inference_output = project_root / "data/inference/pred/rgb_dem_extent"
test_images = project_root / "data/inference/images/aerial"
predict_command = (
    f"terratorch predict \
    --config {config_file} \
    --ckpt_path {checkpoint_path} \
    --trainer.default_root_dir {output_dir} \
    --predict_output_dir {inference_output} \
    --data.init_args.predict_data_root.RGB {test_images} \
    --data.init_args.predict_data_root.DEM {test_images} "
    ""
)
print(f"{command=}")
# os.system(predict_command)

# visualisation comparing s2 vs rgb+dem model
...
