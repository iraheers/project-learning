#!/bin/bash

while read dish_id; do
  echo "Downloading $dish_id..."

  # Download side-angle videos
  gsutil -m cp -r \
    gs://nutrition5k_dataset/nutrition5k_dataset/imagery/side_angles/$dish_id \
    ./nutrition5k_dataset/imagery/side_angles/

  # Download overhead RGB-D images
  gsutil -m cp -r \
    gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/$dish_id \
    ./nutrition5k_dataset/imagery/realsense_overhead/

done < sample_dish_ids.txt
