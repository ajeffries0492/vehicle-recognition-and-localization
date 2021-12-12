# USEFUL COMMANDS

## Train Model
```bash
python model_main_tf2.py --model_dir=models/three_class_resnet50_v1_fpn_120821 --pipeline_config_path=models/three_class_resnet50_v1_fpn_120821/pipeline.config
```
## Evaluate Hold Out Test Data
```bash
python model_main_tf2.py --model_dir=models/three_class_resnet50_v1_fpn_120821 --pipeline_config_path=models/three_class_resnet50_v1_fpn_120821/pipeline.config --checkpoint_dir=models/three_class_resnet50_v1_fpn_120821
# Check on Model using TensorBoard
tensorboard --logdir=models/three_class_resnet50_v1_fpn_120821
```
## Save Model
```bash
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\three_class_resnet50_v1_fpn_120821\pipeline.config --trained_checkpoint_dir .\models\three_class_resnet50_v1_fpn_120821\ --output_directory .\exported-models\three_class_resnet50_v1_fpn_120821
```
