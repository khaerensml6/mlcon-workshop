# 🍫 Building a Controlnet pipeline for interior design with Fondant

This example demonstrates an end-to-end fondant pipeline to collect and process data for the fine-tuning of a [ControlNet](https://github.com/lllyasviel/ControlNet) model, focusing on images related to interior design.

## Pipeline overview

There are 5 components in total, these are:

1. [**Prompt Generation**](components/generate_prompts): This component generates a set of seed prompts using a rule-based approach that combines various animals, clothing items, and media such as image, sketch and painting. You're free to change the prompts to your liking.

2. [**Image URL & Caption Retrieval**](https://github.com/ml6team/fondant/tree/main/components/prompt_based_laion_retrieval): This component retrieves images from the [LAION-5B](https://laion.ai/blog/laion-5b/) dataset based on the seed prompts. The retrieval itself is done based on CLIP embeddings similarity between the prompt sentences and the captions in the LAION dataset. This component doesn’t return the actual images yet, only the URLs and captions. A later component in the pipeline will then download these images.

3. [**Filter on caption artefacts**] Using the fondant dataset inspection tool you can visualize the captions and filter out the ones that are not relevant. This is done by adding a filter component to the pipeline. It's up to you to see which criteria we need to use to filter captions. For example, you can filter out captions that contain very long strings, or very short ones. Also email addresses and non-alphanumeric characters can be filtered out.


4. [**Download Images**](https://github.com/ml6team/fondant/tree/main/components/download_images): This component downloads the actual images based on the URLs retrieved by the previous component. It takes in the URLs as input and returns the actual images, along with some metadata (like their height and width).


5. [**Create Conditionings**] This component creates the conditionings for the ControlNet model. It takes in the laion images as input and returns the conditionings. It's up to you to decide which conditionings you want to use, but unless you're running on a GPU, it's adviced to stick to classical computer vision processing like:
- canny edges
- pixelated version of input
- Smoothed out H, S, V channels
- contours
- blob detection

## Environment

Please check that the following prerequisites are:
- A python version between 3.8 and 3.10 is installed on your system
  ```shell
  python --version
  ```
- Docker compose is installed on your system and the docker daemon is running
  ```shell
  docker compose version
  docker info
  ```
- Fondant is installed
  ```shell
  fondant
  ```

## Implementing the pipeline

The pipeline is already partially implemented in [pipeline.ipynb](pipeline.ipynb).
Please have a look at the notebook and walk through the steps.

The components that need your attention are:
1. **the filter component**
2. **the conditioning component**

## Inspecting the dataset

Once you've implemented the pipeline, you can inspect the dataset using the fondant dataset inspection tool.
This can be done by executing the last cell of the notebook, and opening http://localhost:5601 in your browser.

If you're happy with your dataset, it's time to scale up. Check
[our documentation](https://fondant.ai/en/latest/pipeline/#compiling-and-running-a-pipeline) for
more information about the available runners.
