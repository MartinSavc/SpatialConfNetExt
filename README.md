# Environment configuration

A sufficient anaconda environment configuration was exported to conda_env_spec.txt.

This can be replicated using:
```
conda create --name myenv --file spec-file.txt
```

# Importing the dataset

To train, evaluate or preview the model, images from the dataset must be extracted to 1_data/1_images/. The names of these images must corespond with the landmark anotations in 1_data/2_labels/ and  data splits in 1_data/1_images/split/.

# Train

To train move to the 2_tensorflow directory and run:

```
python spatial_conf_net_v52_exp.py train -e 150
```

After training the model is stored in 2_tensorflow/spatial-config-net-v52-models/model_2022_01_01-08_00_00/. The last part of the folder name will be the date and time of training.

# Preview 

To preview this model run:
```
python spatial_conf_net_v52_exp.py preview --best --model spatial-config-net-v52-models/model_2022_01_01-08_00_00/
```

This will start an preview application with a cmd interface. Entering "train" or "test" will preview the next training or testing image, displayed in a separate window with clickable cephalometric points.

# Evaluate

To evaluate the model run:
```
python spatial_conf_net_v52_exp.py evaluate --best --model spatial-config-net-v52-models/model_2022_01_01-08_00_00/
```

# Analyze the results
This will evaluate the model on all images, the generated results will be stored in 3_analysis/3_npys/spatial-config-net-v52_models/model_2022_01_01-08_00_00/.

To generate an analysis of the results for all test images, from the 3_analysis/ directory run:
```
python 5_benchResults.py -P 3_npys/spatial-config-net-v52_models/model_2022_01_01-08_00_00/model_best/ -L ../1_data/1_images/split/testAllFilesList-bench.txt 
```





