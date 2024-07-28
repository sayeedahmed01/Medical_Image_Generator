# Medical_Image_Generator

### Requirements
Generate common skin rashes based on textual commands varying the following three factors:

1) Skin rash type (e.g., eczema, ringworm, dermatitis)

2) Skin color (e.g., fair, brown, black)

3) Affected area (e.g., chest, neck, hand)

An example command will be like "generate a few images of a ringworm type of rash at the back of the neck area on a fair skin". Build an interface and a deep generative model to process such queries and visualize the output. Please deal with 3-4 rash types that you are not uncomfortable to look at.

Hint: Explore using latent diffusion model, fine-tuning its CLIP model component. Make sure to collect some training images from the internet.