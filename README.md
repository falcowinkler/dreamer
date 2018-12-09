# dreamer
Generate tile data for levels in the game n.

Format conversion and rendering: https://github.com/falcowinkler/copicat

The dataset is seperate: https://github.com/falcowinkler/n-maps-dataset.

This script generates protobuf files every some iterations,
they can be converted into images with copicat.

### network architecture

TODO

Generating discrete data is apparently a hard problem, some GAN's have been developed that can handle discrete data such as BGAN's
https://www.microsoft.com/en-us/research/blog/boundary-seeking-gans-new-method-adversarial-generation-discrete-data/
Another method is sequential generation.

Well, for now, here is a sample from iteration 0 with vanilla gan 🎉: 
 <p align="center">
  <img src="https://github.com/falcowinkler/dreamer/raw/master/docu/sample.png" width="350">
</p>
