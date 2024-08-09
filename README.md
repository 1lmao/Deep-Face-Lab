# DeepFaceLab Project

Followed the online tutorial on https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/ in order to make deep fake video. These were the steps taken. PLEASE DO TAKE NOTE of installing the appropriate libraries refer to : Troubleshooting GPU Detection section for this. We ran into this problem ourselves and it caused a lot of problems. At some point the process timed out so please refer to Note on GPU Process Time in order to make the appropriate changes so the process keeps running.

## System Configuration

- **OS:** Linux Mint
- **GPU:** NVIDIA RTX 3090 (24GB)
- **RAM:** 16GB

## Steps Followed

### 1. Setup

#### Install Anaconda

- Follow the standard Anaconda installation steps.

#### Install DeepFaceLab

1. **Check for Compatible Versions:**
   - Check the latest cuDNN and CUDA Toolkit versions for your GPU device at [TensorFlow Tested Build Configurations](https://www.tensorflow.org/install/source#tested_build_configurations).
   - Check your CUDA version and then search for compatible versions using:
     ```sh
     conda search cudnn
     conda search cudatoolkit
     conda search tensorflow-gpu
     ```

2. **Create Conda Environment:**
   - Create a Conda environment with the compatible versions you identified:
     ```sh
     conda create -n deepfacelab -c main python=[version] cudnn=[version] cudatoolkit=[version] tensorflow-gpu=[version]
     ```
   - Example:
     ```sh
     conda create -n deepfacelab -c main python=3.7 cudnn=7.6.5 cudatoolkit=10.2.89 tensorflow-gpu=2.4.1
     ```

3. **Activate Environment and Clone Repositories:**
   - Activate the Conda environment:
     ```sh
     conda activate deepfacelab
     ```
   - Clone the DeepFaceLab repositories:
     ```sh
     git clone --depth 1 https://github.com/nagadit/DeepFaceLab_Linux.git
     cd DeepFaceLab_Linux
     git clone --depth 1 https://github.com/iperov/DeepFaceLab.git
     ```
   - Install required Python packages:
     ```sh
     python -m pip install -r ./DeepFaceLab/requirements-cuda.txt
     ```

4. **Set Up cuDNN and Toolkit Paths:**
   - Add the cuDNN and CUDA Toolkit paths in your `.sh` file. Open the bash configuration file and copy and paste the paths.

#### Verify TensorFlow GPU Detection

```sh
conda activate deepfacelab
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### Troubleshooting GPU Detection
If you encounter issues with face detection in Step 4, (faces not being detected and high RAM usage), it might be because TensorFlow isn't recognizing your GPU. To fix this follow these steps:

1. Locate your CUDA library path. Make sure you have this installed otherwise the model will not run properly reference line above.

```sh
#See point 2 line 1 for path setup in bashrc after finding the path
cd /
echo $PATH | sed "s/:/\n/g" | grep "cuda/bin" | sed "s/\/bin//g" |  head -n 1 


#See point 2 line 2 for path setup in bashrc after finding the path
cd ~/
find -iname libcudnn.so*
```

   
2. Add the following path to your bash file ( cd home, vi .bashrc), adjusting the paths to match your system:
Eg.
```bash
1 export PATH=/lib/cuda/bin:$PATH
2 export LD_LIBRARY_PATH=/home/[user]/anaconda3/lib:$LD_LIBRARY_PATH
```
These lines ensure that the CUDA binaries are in your PATH and that the necessary libraries are accessible to TensorFlow.



### 2. Download Pre trained model

[Download Pretained Dataset](Add your cmpressed link here)

```sh
    cd DeepFaceLab_Linux/DeepFaceLab
    mkdir pretrain_CelebA # Keep folder name the same because the code directly references this folder
    cd pretrain_CelebA
    # Place the downloaded compressed dataset of your choice here

    #if tar file

    tar -xvf filename.tar.gz

    #If zip file

    unzip filename.zip
```



### 3. Execution

  ```sh
  cd DeepfaceLab_Test/scripts
  ```

  #### Step 1: [Clear Workspace & Import Data](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/#step-1-clear-workspace-import-data)


  ##### Optional: 1) clear workspace.bat
  ```sh
  ./1_clear_workspace.sh
  ```

  #### Place source and destination video in workspace folder with below filename format
    
  ```sh
  data_src.mp4
  data_dst.mp4
  ```


  #### [Step 2: Extract Source Frame Images from Video](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/#step-3-extract-destination-frame-images-from-video)
  Run below command
  ```sh
  ./2_extract_image_from_data_src
  ```

  #### [Step 3: Extract Destination Frame Images from Video](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/#step-3-extract-destination-frame-images-from-video)
  Run below command

  ```sh
  ./3_extract_image_from_data_dst.sh
  ```

  ##### Optional: 3. denoise data_dst images
  Run below command
  ```sh
  ./3.1_denoise_data_dst_images.sh
  ```

  #### [Step4: Extract Source Faceset](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/#step-4-extract-source-faceset)

  Run below command
  ```sh
  ./4_data_src_extract_faces_S3FD.sh
  ```
  ##### Note on GPU Process Time
  
  During the execution of Step 4, you may encounter issues where the GPU process takes a long time and gets killed. To resolve this, modify the following file:
  
  `DeepFaceLab_Linux/DeepFaceLab/core/joblib/SubprocessorBase.py`
  
  Change lines 103-105 as follows:
  
  ```python
  103         self.SubprocessorCli_class = SubprocessorCli_class
  104         #self.no_response_time_sec = no_response_time_sec  # Disabled
  105         self.no_response_time_sec = 0  # Added
  ```

  ##### [Step 4.1: View Source Faceset Result](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/#step-4-1-view-source-faceset-result)

  
  Go to `DeepFaceLab_Linux/workspace/data_src/aligned` path and verify if faces are extracted accurately/


  ##### [Step 4.2: Source Faceset Sorting & Cleanup](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/#step-4-2-source-faceset-sorting-cleanup)
  Run below commnad

  ```sh
  ./4.2_data_src_sort.sh
  ```

  Go to `DeepFaceLab_Linux/workspace/data_src/aligned` path and delete all images having no face/

  

  #### [Step 5: Extract Destination Faceset](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/#step-5-extract-destination-faceset)
  Run below command
  ```sh
  ./5_data_dst_extract_faces_S3FD.sh
  ```

  ##### [Step 5.1: View Destination Faceset Result](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/#step-5-1-view-destination-faceset-result)

  
  Go to `DeepFaceLab_Linux/workspace/data_dst/aligned` path and verify if faces are extracted accurately


  ##### [Step 5.2: Source Faceset Sorting & Cleanup](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/#step-5-2-destination-faceset-sorting-cleanup-re-extraction)
  Run below commnad

  ```sh
  ./5.2_data_dst_sort.sh
  ```

  Go to `DeepFaceLab_Linux/workspace/data_dst/aligned` path and delete all images having no face/

  
  ##### [Step 5.3: XSeg Mask Labeling & XSeg Model Training](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/#step-5-3-xseg-mask-labeling-xseg-model-training)


  Run below commnads

  ```sh
  
  ./5_XSeg_generic_wf_data_dst_apply.sh
  ./5_XSeg_data_dst_mask_edit.sh
  ./5_XSeg_data_dst_mask_fetch.sh
  ./5_XSeg_data_dst_mask_remove.sh
  ./5_XSeg_data_dst_mask_apply.sh
  ./5_XSeg_data_dst_trained_mask_remove.sh

  ./5_XSeg_generic_wf_data_src_apply.sh
  ./5_XSeg_data_src_mask_edit.sh
  ./5_XSeg_data_src_mask_fetch.sh
  ./5_XSeg_data_src_mask_remove.sh
  ./5_XSeg_data_src_mask_apply.sh
  ./5_XSeg_data_src_trained_mask_remove.sh

  ./5_XSeg_train
  ./5_XSeg_generic_wf_data_dst_apply.sh
  ./5_XSeg_generic_wf_data_src_apply.sh
  ```


  ##### [Step 6: Deepfake Model Training](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/#step-6-deepfake-model-training)

  Used SAEHD - Sparse Auto Encoder HD. The standard model and trainer for most deepfakes.
  Execute below command and check on link to understand Model Training Settings. The training should be done in 4 phases of mininmum 100K each. After reaching 100k just `ctrl+c` or `enter` in GUI.


###### Phase 1: Pretraining (Optional)

> Step 1 – Pretrain the Model or Import a Pretrained Model

- Enter all the model settings.
- Enable Pretrain mode.

###### Phase 2: Generalization / Warped Training

> Step 2 – Random Warp

- Enable Random Warp of samples.
- Enable Masked training (WF/Head only).
- Disable True Face Power (DF models only).
- Disable GAN.
- Disable Pretrain mode.
- **Optional:** 
  - Flip SRC randomly.
  - Flip DST randomly.
  - Color Transfer Mode.
  - Random HSL.
  - Add or remove faceset images and alter masks during warp phase.
  - Enable gradient clipping as needed.

>  Step 3 – Eyes and Mouth Priority (Optional)

- Enable Eyes and mouth priority.

> Step 4 – Uniform Yaw (Optional)

- Disable Eyes and mouth priority.
- Enable Uniform yaw distribution of samples.

> Step 5 – Learning Rate Dropout (Optional)

- Enable Use learning rate dropout.
- **Optional:** Disable Uniform yaw distribution of samples.

###### Phase 3: Normalization / Regular Training

>  Step 6 – Regular Training

- Disable Random Warp.
- Disable Uniform Yaw.
- Disable Eyes and mouth priority.
- Disable Use learning rate dropout.

>  Step 7 – Style and Color (Optional)

- Enable Blur out mask.
- Enable ‘True Face’ power (DF only).
- Enable Face style power.
- Enable Background Style Power.

> Step 8 – Eyes and Mouth Priority (Optional)

- Enable Eyes and mouth priority.

> Step 9 – Uniform Yaw (Optional)

- Disable Eyes and mouth priority.
- Enable Uniform yaw distribution of samples.

> Step 10 – LRD (Optional)

- Enable Use learning rate dropout.
- Disable Eyes and mouth priority.
- **Optional:** Disable Uniform yaw distribution of samples.

###### Phase 4: Enhancement / GAN Training (Optional)

> Step 11 – GAN
(Model settings:)
- Disable Eyes and mouth priority.
- Disable Uniform yaw distribution of samples.
- Set GAN power.
- Set GAN patch size.
- Set GAN dimensions.


#### [Step 7: Merge Deepfake Model to Frame Images](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/#step-7-merge-deepfake-model-to-frame-images)
Run below commands, check on the link to understand mergining

```sh
./7_merge_SAEHD.sh
```

#### [Step 8: Merge Frame Images to Video](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/#step-8-merge-frame-images-to-video)
Run below command
```sh
./8_merged_to_mp4.sh
```

#### [Step 9: View Result Video](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/#step-9-view-result-video)

Go to `DeepFaceLab_Linux/workspace/` and play `result.mp4`





[Src](https://www.deepfakevfx.com/guides/deepfacelab-2-0-guide/)
