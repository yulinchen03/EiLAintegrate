# Investigating Cross-cultural Generalizability of Facial Emotion Recognition with Multi-dataset Training
## Yulin Chen, University of Twente 2024
Facial emotion recognition (FER) is among computer vision’s most complex
fields and has several practical uses in human-computer interaction (HCI)
and psychology. Currently, existing FER models are trained on datasets
dominated by a singular ethnicity. As a result, the accuracy is often limited when the model is deployed in
the real world, where the population is much more culturally diverse. This
research will investigate the impact of augmenting existing FER datasets
with EiLA (Emotions in LatAm Dataset), a newly curated emotion recog-
nition in-the-wild dataset consisting of video recordings of Latin American
populations and their facial expressions, on the accuracy and performance of
well-known FER models. 

![](https://github.com/yulinchen03/EiLAintegrate/blob/master/Misc_images_for_thesis/difference_after_integration.png?raw=true)
![](https://github.com/yulinchen03/EiLAintegrate/blob/master/Misc_images_for_thesis/FER.jpg?raw=true)

## Navigation
1. Pre-processing: To check out how pre-processing was done on the EiLA dataset, check [crop_faces.py](https://github.com/yulinchen03/EiLAintegrate/blob/master/EiLA/Preprocessing/1.%20Detect_and_Crop_Images/crop_faces.py) & [process_image.py](https://github.com/yulinchen03/EiLAintegrate/blob/master/EiLA/Preprocessing/2.Resized_image/process_image.py)
2. Data Integration: To look at how data was integrated between datasets, checkout [dataset_integration](https://github.com/yulinchen03/EiLAintegrate/blob/master/Experiment%20Notebooks/dataset_integration.ipynb)
3. Training & Testing: To see how training and testing were performed, checkout [Benchmark_experiments](https://github.com/yulinchen03/EiLAintegrate/tree/master/Experiment%20Notebooks/Benchmark_experiments), [FER2013_experiments](https://github.com/yulinchen03/EiLAintegrate/tree/master/Experiment%20Notebooks/FER2013_experiments) and [SFEW_experiments](https://github.com/yulinchen03/EiLAintegrate/tree/master/Experiment%20Notebooks/SFEW_experiments)
4. Evaluation: To see how the model changed its predictions before vs after integrating datasets, checkout [compile_predictions_per_race](https://github.com/yulinchen03/EiLAintegrate/tree/master/Experiment%20Notebooks/compile_predictions_per_race)
NOTE: This will only work if the experimental notebooks have been run


