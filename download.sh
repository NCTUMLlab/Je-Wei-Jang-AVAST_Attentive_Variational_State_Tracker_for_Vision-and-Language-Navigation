# r2r dataset
wget https://www.dropbox.com/s/hh5qec8o5urcztn/R2R_train.json -P tasks/env/r2r_dataset/
wget https://www.dropbox.com/s/8ye4gqce7v8yzdm/R2R_val_seen.json -P tasks/env/r2r_dataset/
wget https://www.dropbox.com/s/p6hlckr70a07wka/R2R_val_unseen.json -P tasks/env/r2r_dataset/
wget https://www.dropbox.com/s/w4pnbwqamwzdwd1/R2R_test.json -P tasks/env/r2r_dataset/
pip3 install gdown
gdown --id 1Wlhp87sjUyhUuVSarrH22VXw7zOVvCfw
mv R2R_data_augmentation_paths.json tasks/env/r2r_dataset/R2R_train_aug.json

# resnet
wget https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip -P tasks/env/img_features/
unzip tasks/env/img_features/ResNet-152-imagenet.zip -d tasks/env/img_features/

# glove
wget https://nlp.stanford.edu/data/glove.6B.zip -P tasks/env/nlp_features/
unzip -j tasks/env/nlp_features/glove.6B.zip glove.6B.300d.txt -d tasks/env/nlp_features/
