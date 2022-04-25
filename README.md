# Augmenting Backbone Approaches To Instance-level Recognition For Artworks
Studying the behaviours and patterns drawn by ResNets on '[The Met](https://www.metmuseum.org/)' dataset.


Here, we use 'The Mini Met' dataset which contains 33501 classes of images. In our experimentation so far, we have delved into
1) [ResNet18 on ImageNet](https://drive.google.com/file/d/1amFEYsUmJkJlG1Kt0RQ_dAiYS-kojrgi/view?usp=sharing): The standard ResNet backbone. 
2) [ResNet18SRC on ImageNet](https://drive.google.com/file/d/1c6X9DxyGKHgKxj69UPZE2BhWvXL2z20X/view?usp=sharing): Trained on Met with contrastive loss (Syn+Real-Closest). Initialization: ImageNet pre-training.
3) [ResNet18-SWSL-SRC](https://drive.google.com/file/d/11aOyuZaUFze7ffDHJz-A7__rWArT2fsW/view?usp=sharing): Trained on Met with contrastive loss (Syn+Real-Closest). Initialization: SWSL.
### Usage

Navigate (```cd```) to ```[YOUR_MET_ROOT]/met/code```. ```[YOUR_MET_ROOT]``` is where you have cloned this repository. 

<details>

  <summary><b>Descriptor extraction</b></summary><br/>
  
  Here, we extract the descriptors of the train, test, and validation sets.

  Run the following to begin extraction of the descriptors for ResNet-18 trained on ImageNet on The Met dataset.
  ```
  python3 extract_descriptors.py
  ```

</details>

<details>

  <summary><b>kNN classifier & evaluation</b></summary><br/>
  
  The next step is to evaluate the performance with GAP and derive accuracies.

  Run the below command and use -h for help options as shown below:
  ```
  python3 -m examples.knn_eval -h
  ```

  Example (using ground truth and descriptors downloaded from [here](http://cmp.felk.cvut.cz/met/), after unzipping both):  
  ```
  python -m examples.knn_eval [YOUR_DESCRIPTOR_DIR] --autotune --info_dir [YOUR_GROUND_TRUTH_DIR]
  ```

</details>

<details>
  
  <summary><b>Training with contrastive loss</b></summary><br/>

  The trained network can be used for descriptor extraction and kNN classification.

  For detailed explanation of the options run:  
  ```
  python3 -m examples.train_contrastive -h
  ```

</details>


---
