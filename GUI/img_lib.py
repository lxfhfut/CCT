import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log, blob_doh
from scipy.ndimage import gaussian_filter
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.segmentation import mark_boundaries
from skimage.morphology import disk, erosion


class ImageProcLib(object):

    def count_dots(self, img, min_sigma=1, max_sigma=3, num_sigma=5, threshold=0.25, threshold_rel=0.5):
        if img.ndim < 3:
            blob_img = np.copy(img)
        else:
            blob_img = img[:, :, 0]  # cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blob_img = gaussian_filter(blob_img, 0.5)
        blobs = blob_log(blob_img, min_sigma=min_sigma,
                         max_sigma=max_sigma,
                         num_sigma=num_sigma,
                         threshold=threshold,
                         threshold_rel=threshold_rel,
                         overlap=1.0,
                         exclude_border=(10))
        # return blob_img, blobs
        if blobs.size == 0:

            return blob_img, np.empty((0, 3))

        circles = blobs.copy()
        circles[:, 2] *= 8
        scores = blob_img[circles[:, 0].astype(np.int), circles[:, 1].astype(np.int)]
        nms_circles = self.nms_circles(circles, scores, 0.1)

        nms_circles[:, 2] /= 8

        return blob_img, nms_circles

    def count_cells(self, model, img, prob_thresh=0.6, nms_thresh=0.5):
        if img.ndim < 3:
            cell_img = np.copy(img)
        else:
            cell_img = img[:, :, 2]  # cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        cell_img = gaussian_filter(cell_img, 3)

        label_img, _ = model.predict_instances(normalize(cell_img), prob_thresh=prob_thresh, nms_thresh=nms_thresh)

        result_img = np.zeros_like(label_img)
        mask = label_img > 0  # label 0 for background
        eroded = erosion(mask, disk(1))
        result_img[eroded] = label_img[eroded]
        return result_img

    def nms_circles(self, circles, scores, iou_threshold):
        # Sort circles by their scores in descending order
        indices = np.argsort(scores)[::-1]
        circles = circles[indices]
        scores = scores[indices]

        # Initialize a list to store the selected circles
        selected_circles = []

        while len(circles) > 0:
            # Select the circle with the highest score
            current_circle = circles[0]
            selected_circles.append(current_circle)

            # Compute the overlap between the selected circle and the remaining circles
            remaining_circles = circles[1:]
            ious = self.compute_ious(current_circle, remaining_circles)

            # Remove circles with overlap greater than the threshold
            circles_to_remove = np.where(ious > iou_threshold)[0]
            circles = np.delete(remaining_circles, circles_to_remove, axis=0)
            scores = np.delete(scores, circles_to_remove)

        return np.array(selected_circles)

    def compute_ious(self, circle, other_circles):
        ious = []
        for idx in range(other_circles.shape[0]):
            circle2 = other_circles[idx, :]
            ious.append(self.circle_iou(circle, circle2))

        return np.array(ious)

    def circle_iou(self, circle1, circle2):
        """
        Calculates intersection over union between two circles.

        Parameters:
            circle1 (tuple): Tuple containing (x,y,r) coordinates of the first circle.
            circle2 (tuple): Tuple containing (x,y,r) coordinates of the second circle.

        Returns:
            iou (float): Intersection over union between the two circles.
        """
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2

        # Calculate the distance between the centers of the two circles
        d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # If the circles don't intersect at all, return 0
        if d > r1 + r2:
            return 0

        # If one circle is completely inside the other, return the ratio of two circle areas
        if d < abs(r1 - r2):
            return min(r1, r2) ** 2 / (max(r1, r2) ** 2)

        # Calculate the areas of the two circles and the overlap
        area1 = math.pi * r1 ** 2
        area2 = math.pi * r2 ** 2
        # angle1 = math.acos((r1 ** 2 + d ** 2 - r2 ** 2) / (2 * r1 * d))
        # angle2 = math.acos((r2 ** 2 + d ** 2 - r1 ** 2) / (2 * r2 * d))
        # overlap = r1**2*angle1 + r2**2*angle2 - 0.5*math.sqrt((r1+r2-d)*(r1+d-r2)*(d-r1+r2)*(d+r1+r2))

        angle1 = 2 * math.acos((r1 ** 2 + d ** 2 - r2 ** 2) / (2 * r1 * d))
        angle2 = 2 * math.acos((r2 ** 2 + d ** 2 - r1 ** 2) / (2 * r2 * d))
        overlap = 0.5 * (angle1 * r1 ** 2 - r1 ** 2 * math.sin(angle1) + angle2 * r2 ** 2 - r2 ** 2 * math.sin(angle2))

        # Calculate and return the intersection over union
        return overlap / (area1 + area2 - overlap)

    # def add_labels(self, img, label_img, color=(1, 1, 0), background_label=0):
    #     rows, cols = label_img.shape[0], label_img.shape[1]
    #     img_with_bnd = mark_boundaries(img, label_img, color, mode='thick', background_label=background_label)
    #     img_with_bnd = (img_with_bnd * 255).astype(np.uint8)
    #     unique_labels = list(np.unique(label_img[:]))
    #     for label_idx, label in enumerate(unique_labels):
    #         if label == background_label:
    #             continue
    #         coordinates = np.argwhere(label_img == label)
    #         center = np.max(coordinates, axis=0)
    #         if center[0] >= rows:
    #             center[0] = rows - 1
    #         if center[1] >= cols:
    #             center[1] = cols - 1
    #         center = center[::-1]
    #         img_with_bnd = cv2.putText(img_with_bnd, str(label_idx+1),
    #                                    tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
    #                                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    #     num_cells = np.unique(label_img[:]).size
    #     return img_with_bnd, num_cells