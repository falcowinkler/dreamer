# dreamer
Generate tile data for levels in the game n.

Format conversion and rendering: https://github.com/falcowinkler/copicat

The dataset is seperate: https://github.com/falcowinkler/n-maps-dataset.

This script generates protobuf files every some iterations,
they can be converted into images with copicat.

### network architecture

The inputs are of shape 31 * 23 * 33 (33 Tiles, one-hot encoded).
8 * 7 * 33.

### results
not much, for now.
Still TODO:
 - sophisticated network architecture
 - automate deployment and results monitoring
 - filter dataset for empty tilesets, filter out levels with certain tags
 
 Heres a sample from iteration 0 🎉: 
 <p align="center">
  <img src="https://github.com/falcowinkler/dreamer/raw/master/docu/sample.png" width="350">
</p>
