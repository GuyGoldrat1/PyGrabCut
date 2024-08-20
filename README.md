
# GrabCut Image Segmentation

This project implements image segmentation using the GrabCut algorithm in Python. The GrabCut algorithm is a method for foreground extraction, typically used to isolate the subject of an image from its background.

## Features

- **Image Segmentation:** Isolate the foreground of an image using the GrabCut algorithm.
- **Interactive Foreground Extraction:** Manually refine the segmentation by defining boxes through a text file.

## Implementation

The implementation is based on the research paper "GrabCut: Interactive Foreground Extraction using Iterated Graph Cuts" by Carsten Rother, Vladimir Kolmogorov, and Andrew Blake. This algorithm provides an efficient and interactive method for segmenting the foreground from the background in an image by iteratively refining a segmentation mask.

## Results
Below are some examples of image segmentation using the GrabCut algorithm:

### Example 1: Banana
Input Image:
banana1.jpg
![alt text]https://github.com/GuyGoldrat1/PyGrabCut/blob/main/inputData/imgs/banana1.jpg
Bounding Box:
(16, 20, 620, 436)

Resulting Image:

Example 2: llama
Input Image:
llama.jpg
Bounding Box:
(112, 106, 370, 371)



## Functions

The main functions in this project include:

### grabcut(img, rect, n_iter=5)`
This is the core function that implements the GrabCut algorithm. It performs the following steps:
- Initializes a mask for the image.
- Segments the foreground using Gaussian Mixture Models (GMMs).
- Iteratively refines the segmentation using graph cuts until convergence or a maximum number of iterations.

### initalize_GMMs(img, mask, n_components=5)`
This function initializes the GMMs for the background and foreground by clustering the pixels within the specified bounding box.

### update_GMMs(img, mask, bgGMM, fgGMM)`
Updates the GMMs based on the current segmentation mask. This step recalculates the parameters of the GMMs to improve segmentation accuracy.

### calculate_mincut(img, mask, bgGMM, fgGMM)`
Performs the graph cut step by constructing a graph from the image and solving the min-cut problem to separate foreground and background.

### update_mask(mincut_sets, mask)`
Updates the segmentation mask based on the result of the min-cut. Pixels are classified as either background or foreground.

---

## Usage

The `grabcut.py` script can be executed with various command-line arguments to control its behavior. Below are the available arguments:

- `--input_name`: Name of the image from the course files (default: `banana1`).
- `--eval`: Whether to calculate metrics such as accuracy and Jaccard index (default: `1`).
- `--input_img_path`: Path to your own image file (default: `''`). If not provided, the script will use an image from the course files.
- `--use_file_rect`: Whether to read the bounding box (rectangle) from the course files (default: `1`). If set to `0`, the `--rect` argument must be used to provide custom rectangle coordinates.
- `--rect`: Custom bounding box coordinates in the format `x,y,w,h` (default: `1,1,100,100`).

### Example Command

Here's how to run the script with the default settings:

```bash
python grabcut.py --input_name banana1
```

If you want to use a custom image and specify a custom bounding box, you can use:

```bash
python grabcut.py --input_img_path path/to/your/image.jpg --use_file_rect 0 --rect 50,50,200,200
```

To evaluate the segmentation quality against ground truth, ensure that the `--eval` argument is set to `1` (which is the default):

```bash
python grabcut.py --input_name banana1 --eval 1
```

