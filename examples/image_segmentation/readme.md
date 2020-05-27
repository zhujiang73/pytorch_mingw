# Image segmentation (The Caltech-UCSD Birds-200-2011 Dataset) 


setup dataset image path:

    edit imgs_comp_mask.py

        imgs_dir = "e:/cv_datasets/CUB_200_2011/images"
        imgs_mask_dir = "e:/cv_datasets/CUB_200_2011/segmentations"


Pretreatment:

    win10  cmd  console :
        
        mk_imgs_dirs.cmd
        imgs_comp_mask.py


training models:

    win10  cmd  console :

        python   gtk_mask_train_quick.py
        .........
        python   gtk_mask_train_continue.py
        .........


