This Repo does not include the model weights.  
This Repo used a preexisting pvc.

To run this model, create a the pvc with the same name as those defined in the train-rayjob.yaml file, or modify the train-rayjob.yaml file to have the correct pvc.

The fine-tune-pvc should have a directory called fine-tuning/Data.  If not, you can change SHARED_STORAGE in the train-rayjob.yaml to valid value. 
The fine-tune-pvc should also contain the file finetunedataset_v2_prompt_tags_removed.csv in fine-tuning/Code/finetunedataset_v2_prompt_tags_removed.csv. You can also change this to appropriate values

The prod-mctssagpt-model should have models or you can change it and change the PRETRAINED_BASE_MODEL variable in the train-rayjob.yaml to match

Use the deploy/build/rayjob.dockerfile to build the rayproject/ray:2.34.1-gpu image