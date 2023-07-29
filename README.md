# Face Mask

The code has been tested with google colab with GPU runtime:
https://colab.research.google.com/drive/1iqP9VVbQztjwsJtVZ01Brao6L36HOdbk

The AI models `model2017-1_face12_nomouth.h5` has been requested from from https://faces.dmi.unibas.ch/bfm/bfm2017.html and` albedoModel2020_face12_albedoPart.h5` from https://github.com/waps101/AlbedoMM/releases/download/v1.0/albedoModel2020_face12_albedoPart.h5. These two files are inside the baselMorphableModel directory. These AI models weights are propretiary IP that cannot be shared, **please see licensing information to know how to use them.**

As in the main branch:

- create a new virtual environment for this branch
- activate the virtual environment
- install the dependencies in the requirements.txt file
- Run the code as demonstrated in the colab above

The masked face and other related data can be found in the output folder.

For detailed used, please refer to the github repository: https://github.com/abdallahdib/NextFace/tree/main

The neccessary information from the snapshot is as follows:

# How to use

## Reconstruction from a single image

- to reconstruct a face from a single image: run the following command:
  `python optimizer.py --input path-to-your-input-image --output output-path-where-to-save-results`

## Reconstruction from multiple images (batch reconstruction)

In case you have multiple images with same resolution, u can run a batch optimization on these images. For this, put all ur images in the same directory and run the following command:
`python optimizer.py --input path-to-your-folder-that-contains-all-ur-images --output output-path-where-to-save-results`

## Reconstruction from mutliple images for the same person

if you have multiple images for the same person, put these images in the same folder and run the following command:

`python optimizer.py --sharedIdentity --input path-to-your-folder-that-contains-all-ur-images --output output-path-where-to-save-results`
the sharedIdentity flag tells the optimizer that all images belong to the same person. In such case, the shape identity and face reflectance attributes are shared across all images. This generally produces better face reflectance and geometry estimation.

Configuration File
The file `optimConfig.ini` allows to control different

- change `device` to 'cpu' to allow for cpu usage

N.B:
** To Generate an output takes 4~5 minutes depending on your GPU performance and longer for CPU.**

#### The output of the optimization is the following:

- render\*{imageIndex}.png: contains from left to right: input image, overlay of the final reconstruction on the input image, the final reconstruction, diffuse, specular and roughness maps projected on the face.
- diffuseMap\*{imageIndex}.png: the estimated diffuse map in uv space
- specularMap\*{imageIndex}.png: the estimated specular map in uv space
- roughnessMap\*{imageIndex}.png: the estimated roughness map in uv space
- mesh{imageIndex}.obj: an obj file that contains the 3D mesh of the reconstructed face
