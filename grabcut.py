import numpy as np
import cv2
import argparse
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import igraph as ig
import os

os.environ['LOKY_MAX_CPU_COUNT'] = '2'




GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8) ## mask the same size as picture 480x640
    mask.fill(GC_BGD) ## file mask with zeros == Hard BG
    x, y, w, h = rect

    # Convert from absolute cordinates
    w -= x
    h -= y

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD  # inside the rectangle soft FG
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD #one point of Hard FG


    bgGMM, fgGMM = initalize_GMMs(img, mask)
    
    # --- Added code
    prev_mask= mask.copy()
    changed_pixels = []
    # --- Added code

    num_iters = 50
    for i in range(num_iters):

        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)


        if check_convergence(mask, prev_mask, changed_pixels):
            break

        prev_mask = mask.copy()

    #Create Final mask
    mask[mask==GC_PR_BGD] = GC_BGD
    mask[mask==GC_PR_FGD] = GC_FGD

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask, n_components=5): 

    # Filtering FG and BG pixels
    bg_pixels = img[(mask == GC_BGD) | (mask == GC_PR_BGD)].reshape(-1, 3)
    fg_pixels = img[(mask == GC_FGD) | (mask == GC_PR_FGD)].reshape(-1, 3)
    
    #Initalizeing GMM using kmeans
    bgGMM = GaussianMixture(n_components=n_components, covariance_type='full',n_init=1, max_iter=100,init_params='kmeans')
    fgGMM = GaussianMixture(n_components=n_components, covariance_type='full',n_init=1, max_iter=100,init_params='kmeans')

    bgGMM.fit(bg_pixels)
    fgGMM.fit(fg_pixels)
    
    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):

    # Filtering FG and BG pixels
    bg_pixels = img[(mask == GC_BGD) | (mask == GC_PR_BGD)].reshape(-1, 3)
    fg_pixels = img[(mask == GC_FGD) | (mask == GC_PR_FGD)].reshape(-1, 3)

    n_components = bgGMM.n_components 


    # Creating the relevant arrays
    bg_means = np.zeros((n_components, 3))
    fg_means = np.zeros((n_components, 3))

    bg_cov = np.zeros((n_components, 3, 3))
    fg_cov = np.zeros((n_components, 3, 3))
    
    bg_weights = np.zeros(n_components)
    fg_weights = np.zeros(n_components)

    for i in range(n_components):
        pixels = bg_pixels[bgGMM.predict(bg_pixels) == i]
        if len(pixels) > 0:
            bg_means[i] = np.mean(pixels, axis=0)
            bg_cov[i] = np.cov(pixels, rowvar=False)
            bg_weights[i] = len(pixels) / len(bg_pixels)

    for i in range(n_components):
        pixels = fg_pixels[fgGMM.predict(fg_pixels) == i]
        if len(pixels) > 0:
            fg_means[i] = np.mean(pixels, axis=0)
            fg_cov[i] = np.cov(pixels, rowvar=False)
            fg_weights[i] = len(pixels) / len(fg_pixels)



    bgGMM.means_ = bg_means
    bgGMM.covariances_ = bg_cov
    bgGMM.weights_ = bg_weights


    fgGMM.means_ = fg_means
    fgGMM.covariances_ = fg_cov
    fgGMM.weights_ = fg_weights



    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    height, width = img.shape[:2]
    num_pixels = height * width


    source = num_pixels
    sink = num_pixels + 1

    
    # add 8 N links to each PIXEL

    RL_diff = img[:, 1:] - img[:, :-1]
    lowR_upL_diff = img[1:, 1:] - img[:-1, :-1]
    up_down_diff = img[1:, :] - img[:-1, :]
    upR_lowL_diff = img[1:, :-1] - img[:-1, 1:]

    #Caculate beta
    beta = 1 / (2 * (np.sum(np.square(RL_diff))+np.sum(np.square(lowR_upL_diff))+\
        np.sum(np.square(up_down_diff))+np.sum(np.square(upR_lowL_diff))) / (4 * height * width - 3 * height - 3 * width + 2))
    
    
    RL_n = 50 * np.exp(-beta * np.sum(np.square(RL_diff), axis=2))
    lowR_upL_n = (50 / np.sqrt(2)) * np.exp(-beta * np.sum(np.square(lowR_upL_diff), axis=2))
    up_down_n = 50 * np.exp(-beta * np.sum(np.square(up_down_diff), axis=2))
    upR_lowL_n = (50 / np.sqrt(2)) * np.exp(-beta * np.sum(np.square(upR_lowL_diff), axis=2))

    img_indexes=np.arange(height*width, dtype=np.uint32).reshape(height, width)
    edge = []
    capacities = []

    #add N-links

    img_direction_1=img_indexes[:, 1:].reshape(-1)
    img_direction_2=img_indexes[:, :-1].reshape(-1)
    edge.extend(list(zip(img_direction_1, img_direction_2)))
    capacities.extend(RL_n.reshape(-1).tolist())

    img_direction_1=img_indexes[1:, 1:].reshape(-1)
    img_direction_2=img_indexes[:-1, :-1].reshape(-1)
    edge.extend(list(zip(img_direction_1, img_direction_2)))
    capacities.extend(lowR_upL_n.reshape(-1).tolist())

    img_direction_1=img_indexes[1:, :].reshape(-1)
    img_direction_2=img_indexes[:-1, :].reshape(-1)
    edge.extend(list(zip(img_direction_1, img_direction_2)))
    capacities.extend(up_down_n.reshape(-1).tolist())

    img_direction_1=img_indexes[1:, :-1].reshape(-1)
    img_direction_2=img_indexes[:-1, 1:].reshape(-1)
    edge.extend(list(zip(img_direction_1, img_direction_2)))
    capacities.extend(upR_lowL_n.reshape(-1).tolist())

    # Summing up weights for each pixel
    pixel_sums = np.zeros((height, width))
    pixel_sums[:, :-1] += RL_n
    pixel_sums[:, 1:] += RL_n

    # Lower-Right to Upper-Left direction
    pixel_sums[:-1, :-1] += lowR_upL_n
    pixel_sums[1:, 1:] += lowR_upL_n

    # Up-Down direction
    pixel_sums[:-1, :] += up_down_n
    pixel_sums[1:, :] += up_down_n

    # Upper-Right to Lower-Left direction
    pixel_sums[:-1, 1:] += upR_lowL_n
    pixel_sums[1:, :-1] += upR_lowL_n

    # Finding the maximum sum
    MAXk = np.max(pixel_sums)


    HFG_loc=np.where(mask.reshape(-1)==GC_FGD)
    HBG_loc=np.where(mask.reshape(-1)==GC_BGD)
    predict_loc=np.where(np.logical_or(mask.reshape(-1)==GC_PR_BGD ,mask.reshape(-1)==GC_PR_FGD))

    edge.extend(list(zip([sink]*predict_loc[0].size, predict_loc[0])))
    capacities.extend((np.array(bgGMM.score_samples(img.reshape(-1, 3)[predict_loc]))*(-1)).tolist())

    edge.extend(list(zip([source]*predict_loc[0].size, predict_loc[0])))
    capacities.extend((np.array(fgGMM.score_samples(img.reshape(-1, 3)[predict_loc]))*(-1)).tolist())


    # Connect FG to sink -  link with Maxk
    edge.extend(list(zip([sink]*HFG_loc[0].size, HFG_loc[0])))
    capacities.extend([MAXk]*HFG_loc[0].size)

    edge.extend(list(zip([sink]*HBG_loc[0].size, HBG_loc[0])))
    capacities.extend([0]*HBG_loc[0].size)

    edge.extend(list(zip([source]*HFG_loc[0].size, HFG_loc[0])))
    capacities.extend([0]*HFG_loc[0].size)

    edge.extend(list(zip([source]*HBG_loc[0].size, HBG_loc[0])))
    capacities.extend([MAXk]*HBG_loc[0].size)

    # Creating graph updating the edges and capacities
    graph = ig.Graph(directed=False)
    graph.add_vertices(num_pixels + 2)
    graph.add_edges(edge)
    graph.es['capacity']= capacities

    # find mincut
    mincut = graph.st_mincut(sink, source, capacity='capacity')
    mincut_sets = mincut.partition
    energy = mincut.value

    return mincut_sets, energy



def update_mask(mincut_sets, mask):
    height, width = mask.shape
    img_indexes=np.arange(height*width, dtype=np.uint32).reshape(height, width)


    fg_set, bg_set  = mincut_sets
    mask[mask==GC_FGD] = GC_PR_FGD

    to_predict_loc=np.where(np.logical_or(mask == GC_PR_BGD,mask == GC_PR_FGD))
    mask[to_predict_loc] = np.where(np.isin(img_indexes[to_predict_loc], fg_set), GC_PR_FGD, GC_PR_BGD)

    return mask


def check_convergence(mask, prev_mask, changed_pixels):

    perv_bg = prev_mask[prev_mask == GC_PR_BGD].shape[0]
    new_bg = mask[mask == GC_PR_BGD].shape[0]

    changed_pixels.append(perv_bg-new_bg)
    if len(changed_pixels) > 3:
        if changed_pixels[-1] == 0 and changed_pixels[-2] == 0:
            return True
        
    return False




def cal_metric(predicted_mask, gt_mask):
    # Calculate accuracy
    total_pixels = predicted_mask.size
    correctly_labeled_pixels = np.sum(predicted_mask == gt_mask)
    accuracy = correctly_labeled_pixels / total_pixels

    # Jaccard similarity
    intersection = np.sum((predicted_mask == 1) & (gt_mask == 1))
    union = np.sum((predicted_mask == 1) | (gt_mask == 1))
    jaccard = intersection / union if union != 0 else 0

    return accuracy*100, jaccard*100

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))

    img = cv2.imread(input_path)



    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
