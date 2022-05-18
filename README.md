# Augmenting Backbone Approaches To Instance-level Recognition For Artworks
Studying the behaviours and patterns drawn by [ResNets](https://en.wikipedia.org/wiki/Residual_neural_network) on '[The Met](https://www.metmuseum.org/)' dataset.


Here, we use 'The Mini Met' & 'Met' [dataset](http://cmp.felk.cvut.cz/met/) which contains 33501 and 220k+ classes of 'art' respectively. In our experimentation so far, we have delved into the following (click to download the Descriptors):
1) ResNet18 on ImageNet ([Mini](https://drive.google.com/file/d/1z1xlRD9-I55N6xh1pki70D0EArHo6uwB/view?usp=sharing)/[Met](https://drive.google.com/file/d/1CUSdCKndQCsX6zJ6q6VHPJsM5HKIHQKJ/view?usp=sharing))
2) ResNet50 on ImageNet ([Mini](https://drive.google.com/file/d/1-JfnXbbdxokhNhne6s4e88F5Evq2Hoa-/view?usp=sharing)/[Met](https://drive.google.com/file/d/1mfhUqmRCHz2iBeZLHJo-HhFih5X2QUsm/view?usp=sharing))
3) R50SWaV ([Mini](https://drive.google.com/file/d/1ei9nZsUOplOjdJT2Ct_uzeGZ6kjToiXt/view?usp=sharing)/[Met](https://drive.google.com/file/d/1Z5mHEY4CAbAzCy2qc6vJYk2GwuxnmJNK/view?usp=sharing))
4) R50SIN on ImageNet ([Mini](https://drive.google.com/file/d/1-G6RT1bxmkYu8wtrVxQrv7FO4tf9uLNc/view?usp=sharing)/[Met](https://drive.google.com/file/d/1qUma78e2HYckELM1G6TMwmTd8XkJlzpU/view?usp=sharing))
5) R18Sw-Sup ([Mini](https://drive.google.com/file/d/1-O5NMlxCAk4_ohG81XChLObo-VSm7i0K/view?usp=sharing)/[Met](https://drive.google.com/file/d/1N5nrLrKsH1bC9wjXYk2f0BrD63_dfzgq/view?usp=sharing))
6) R50Sw-Sup ([Mini](https://drive.google.com/file/d/1-ZhZGyWArJpna0a6gDC1XTE556YjdfFf/view?usp=sharing)/[Met](https://drive.google.com/file/d/1E1hAa98S-i6l79h_acy96gdkA8GjAW_9/view?usp=sharing))
7) ResNeXt-50-32x4d-SWSL ([Mini](https://drive.google.com/file/d/1-l_a-kqHPzbBvkppCMG7hnQTCJoMoYlm/view?usp=sharing)/[Met](https://drive.google.com/file/d/1-M6H1kPduHafLEm9ImFoTD6Mgvay023k/view?usp=sharing))
8) ResNeXt-101-32x4d-SWSL ([Mini](https://drive.google.com/file/d/1-x6cCo56_cv1YXjs9SWcd4xv_ASfze3q/view?usp=sharing)/[Met](https://drive.google.com/file/d/1kZFIrGgbROUrhZFQNePlM8xgmrXgMzQ5/view?usp=sharing))
9) ResNeXt-101-32x8d-SWSL ([Mini](https://drive.google.com/file/d/1-3_4rTSCmF4BAQTPcD7Yf17a6pGPXZ_S/view?usp=sharing)/[Met](https://drive.google.com/file/d/1GV0jzMkeMvDNBtUdEk4JNvBAToLcUjCI/view?usp=sharing))
10) ResNeXt-101-32x16d-SWSL ([Mini]()/[Met](https://drive.google.com/file/d/1S1My1S9Z7y2ZdXK7HoN1FVVwzv8C9Xft/view?usp=sharing))

### Usage



Navigate (```cd```) to ```[YOUR_MET_ROOT]/met/code```. ```[YOUR_MET_ROOT]``` is where you have cloned this repository. 
<details>
  <summary><b>Readily train a non-parametric model</b></summary><br/>
  
  Here, we collectively perform the training and extract the descriptors for the network variant that you wish to run from this list:<br/>
  r18INgem<br/>
  r50INgem<br/>
  r50_swav_gem<br/>
  r50_SIN_gem<br/>
  r50INgem_caffe<br/>
  r18_sw-sup_gem<br/>
  r50_sw-sup_gem<br/>
  resnext50_32x4d_swsl<br/>
  resnext101_32x4d_swsl<br/>
  resnext101_32x8d_swsl<br/>
  resnext101_32x16d_swsl<br/>
  
  Enter the variant name as one of the above when prompted.<br/>
  For the datasets, you can choose to train it on the Mini dataset or the full dataset. You can download the datasets [here](http://cmp.felk.cvut.cz/met/).
  <br/>You can download the train, test, and validation descriptors [here](http://cmp.felk.cvut.cz/met/).<br/>
  Once ready, run the following:
  ```
  python3 train_the_model.py
  ```
  </details>
  <details>
  <summary><b>Perform a KNN test</b></summary>
  <br/>You can download the train, test, and validation descriptors [here](http://cmp.felk.cvut.cz/met/).<br/>
 Download the train descriptors from the list on top of the README.<br/>
  Once ready, run the following and follow the prompts:
  ```
  python3 run_knn_test.py
  ```
  </details>
  
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

  Train using a parametric approach with contrastive learning.

  For detailed explanation of the options run:  
  ```
  python3 -m examples.train_contrastive -h
  ```

</details>


---
