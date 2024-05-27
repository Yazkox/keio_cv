import cv2
import numpy as np
from typing import List, Tuple
import glob
import scipy.linalg
from math import floor

IMG_DIR = "./stitch_photos"  # Directory of input images


class image_stitcher:
    TRANSLATION = 0
    SIMILARITY = 1
    AFFINE = 2
    HOMOGRAPHY = 3

    def __init__(self, match_amount: int = 100) -> None:
        self.reference_frame = None
        self.ref_image = None
        self.images = []
        self.n = 0
        self.kp_ref = None
        self.desc_ref = None
        self.key_points = []
        self.descriptors = []
        self.matched_keypoints = []
        self.transformation_types = []
        self.transformation_matrixes = []
        self.match_amount = match_amount

    def load_imgs(self, img_dir: str, reference_frame: int = 0, img_format: str = "jpg"):
        """Loads image from the specified directory

        Args:
            img_dir (str): directory of the images
            reference_frame (int, optional): frame number to use as reference image. Defaults to 0.
            img_format (str, optional): file extension of the images. Defaults to "jpg".
        """
        self.reference_frame = reference_frame
        fnames = glob.glob("{}/*.{}".format(img_dir, img_format))
        if reference_frame >= len(fnames):
            raise Exception("Reference frame not in list of images")
        self.images = []
        for i, fname in enumerate(fnames):
            if i == reference_frame:
                self.ref_image = cv2.imread(fname)
            else:
                self.images.append(cv2.imread(fname))
        self.n = len(self.images)

    def get_features(self):
        """Gathers the features of the loaded images using ORB"""
        if len(self.images) == 0:
            raise Exception("Not enough images loaded (min 2)")
        orb = cv2.ORB_create()
        self.key_points, self.descriptors = [], []
        self.kp_ref, self.desc_ref = orb.detectAndCompute(cv2.cvtColor(self.ref_image, cv2.COLOR_RGB2GRAY), None)
        for image in self.images:
            kp, desc = orb.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
            self.key_points.append(kp), self.descriptors.append(desc)

    def match_features(self, display: bool = False, display_scale: float = 1):
        """Matches the features of the images

        Args:
            display (bool, optional): displays the matched features. Defaults to False.
            display_scale (int, optional): scale of the display. Defaults to 1.

        Raises:
            Exception: if the features are not initialized
        """
        if len(self.descriptors) == 0 or self.desc_ref is None:
            raise Exception("Features not initialized")
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.matched_keypoints = []
        for k in range(self.n):
            matches = bf.match(self.desc_ref, self.descriptors[k])
            matches = sorted(matches, key=lambda x: x.distance)[: self.match_amount]
            src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            dst_pts = np.float32([self.key_points[k][m.trainIdx].pt for m in matches]).reshape(-1, 2)
            self.matched_keypoints.append((src_pts, dst_pts))

            if not display:
                continue
            img3 = cv2.drawMatches(
                self.ref_image,
                self.kp_ref,
                self.images[k],
                self.key_points[k],
                matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv2.imshow(
                "Match {}".format(k),
                cv2.resize(img3, (floor(img3.shape[1] * display_scale), floor(img3.shape[0] * display_scale))),
            )

    def find_transformation(self, frame: int, transformation_type: int) -> np.ndarray:
        """Computes the transformation matrixe of the specified frame to the reference image using the matched keypoints

        Args:
            frame (int): index of the frame
            transformation_type (int): type of transformation to use

        Returns:
            np.ndarray: transformation matrix
        """
        Xis = self.matched_keypoints[frame][1]
        Xpis = self.matched_keypoints[frame][0]
        if transformation_type == self.HOMOGRAPHY:
            M, _ = cv2.findHomography(Xis, Xpis)
        else:
            J = lambda x: self.get_J(x, transformation_type)
            A = sum([np.dot(np.transpose(J(Xis[i])), J(Xis[i])) for i in range(len(Xis))])
            b = sum([np.dot(np.transpose(J(Xis[i])), Xpis[i] - Xis[i]) for i in range(len(Xis))])
            p = scipy.linalg.solve(A, b)
            M = self.get_geo_mat(p, transformation_type)
        return M

    def find_transformations(self, transformation_types: List[int]):
        """Computes the transformation matrixes of all the frame to the reference frame

        Args:
            transformation_types (List[int]): list of the transformation type to use for each image

        Raises:
            Exception: when the input list length does not corresponds to the frame count
        """
        if len(transformation_types) != len(self.images):
            raise Exception("Incorrect number of transformation types")
        self.transformation_types = transformation_types
        self.transformation_matrixes = []
        for k in range(len(self.images)):
            self.transformation_matrixes.append(self.find_transformation(k, transformation_type=transformation_types[k]))

    def get_warped_corner(self, frame: int, corner: int) -> Tuple[int, int]:
        """Computes the coordinate of a specific corner of a frame after being warped

        Args:
            frame (int): frame index
            corner (int): corner index starting from top left counter clockwise

        Returns:
            int, int: x, y (x being the height and y the width)
        """
        M = self.transformation_matrixes[frame]
        img_x, img_y = self.images[frame].shape[:2]
        corners = [[0, 0], [img_x, 0], [img_x, img_y], [0, img_y]]
        p = corners[corner]
        if M.shape == (3, 3):
            x, y, f = np.dot(M, np.array([p[1], p[0], 1]))
            return (floor(y / f), floor(x / f))
        x, y = np.dot(M, np.array([p[1], p[0], 1]))
        return (floor(y), floor(x))

    def get_canvas_size(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Computes the canvas size , as well as the frame origin so that every frame once transformed fits into it

        Returns:
            (int, int), (int, int): starting x, starting y, height, width
        """
        x_min, y_min = 0, 0
        x_max, y_max = self.ref_image.shape[:2]
        for frame in range(self.n):
            for corner in range(4):
                x, y = self.get_warped_corner(frame, corner)
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
        return (-x_min, -y_min), (x_max - x_min, y_max - y_min)

    def warp_image(self, frame: int) -> np.ndarray:
        """Warp the image using the computed transformation matrix

        Args:
            frame (int): frame index

        Returns:
            np.ndarray: warped image
        """
        M = self.transformation_matrixes[frame]
        img = self.images[frame]
        dsize = (img.shape[1], img.shape[0])
        if self.transformation_types[frame] == self.TRANSLATION:
            return cv2.warpAffine(img, M, dsize, borderMode=cv2.BORDER_TRANSPARENT)
        elif self.transformation_types[frame] == self.SIMILARITY:
            return cv2.warpAffine(img, M, dsize, borderMode=cv2.BORDER_TRANSPARENT)
        elif self.transformation_types[frame] == self.AFFINE:
            return cv2.warpAffine(img, M, dsize, borderMode=cv2.BORDER_TRANSPARENT)
        elif self.transformation_types[frame] == self.HOMOGRAPHY:
            return cv2.warpPerspective(img, M, dsize, borderMode=cv2.BORDER_TRANSPARENT)

    def update_images(self):
        """Update the images by placing them into the canvas, and updates the features coordinates"""
        (x, y), (height, width) = self.get_canvas_size()
        offset = np.array([y, x])
        blank = np.zeros((height, width, 4), np.uint8)
        self.ref_image = self.place_img(cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2RGBA), blank.copy(), x, y)
        for frame in range(self.n):
            self.images[frame] = self.place_img(cv2.cvtColor(self.images[frame], cv2.COLOR_BGR2RGBA), blank.copy(), x, y)
            for k in range(len(self.matched_keypoints[frame][0])):
                self.matched_keypoints[frame][0][k] += offset
                self.matched_keypoints[frame][1][k] += offset

    def draw(self, transparency: bool = False, scale: float = 1):
        """Draw the stitched images

        Args:
            transparency (bool, optional): draw transparency. Defaults to False.
            scale (float, optional): display scale. Defaults to 1.
        """
        channels = 3 + transparency
        alpha = 1 / (self.n + 1)
        canvas = self.ref_image[:, :, :channels]
        for frame in range(self.n):
            img = self.warp_image(frame)
            if transparency:
                canvas = cv2.addWeighted(canvas, 1 - alpha, img, alpha, 0.0)
            else:
                canvas = np.maximum(canvas, img[:, :, :3])
        if transparency:
            stitch = cv2.cvtColor(canvas, cv2.COLOR_RGBA2RGB)
        else:
            stitch = canvas
        stitch = cv2.resize(stitch, (floor(stitch.shape[1] * scale), floor(stitch.shape[0] * scale)))
        cv2.imshow("Original", stitch)

    def stitch(self, transformations: List[int], display_features: bool = False, display_scale: float = 1):
        """Apply all the steps to stitch the images

        Args:
            transformations (List[int]): list containing which transformation to use with each images
            display_features (bool, optional): display the matched features or not. Defaults to False.
            display_scale (float, optional): scale of the drawn image the . Defaults to 1.
        """
        self.get_features()
        self.match_features(display=display_features, display_scale=display_scale)
        self.find_transformations(transformations)
        self.update_images()
        self.find_transformations(transformations)

    @staticmethod
    def get_geo_mat(p: List[float] | np.ndarray, transformation_type: int) -> np.ndarray:
        """Construct the transformation matrix using the given parameters

        Args:
            p (List[float] | np.ndarray): parameter of the transformation
            transformation_type (int): transformation type

        Raises:
            Exception: if the transformation type is not supported

        Returns:
            np.ndarray: Transformation matrix
        """
        if transformation_type == image_stitcher.TRANSLATION:
            return np.array([[1, 0, p[0]], [0, 1, p[1]]])
        elif transformation_type == image_stitcher.SIMILARITY:
            return np.array([[1 + p[2], -p[3], p[0]], [p[3], 1 + p[2], p[1]]])
        elif transformation_type == image_stitcher.AFFINE:
            return np.array([[1 + p[2], p[3], p[0]], [p[4], 1 + p[5], p[1]]])
        else:
            raise Exception("transformation type '{}' not supported".format(transformation_type))

    @staticmethod
    def get_J(x: List[float] | np.ndarray, transformation_type: int) -> np.ndarray:
        """Computes the jacobian matrix for given x and y and a given transformation type

        Args:
            x (List[float] | np.ndarray): point : x, y
            transformation_type (int): transformation to use

        Raises:
            Exception: when transformation type is not supported

        Returns:
            np.ndarray: jacobian matrix
        """
        if transformation_type == image_stitcher.TRANSLATION:
            return np.identity(2)
        elif transformation_type == image_stitcher.SIMILARITY:
            return np.array([[1, 0, x[0], -x[1]], [0, 1, x[1], x[0]]])
        elif transformation_type == image_stitcher.AFFINE:
            return np.array([[1, 0, x[0], x[1], 0, 0], [0, 1, 0, 0, x[0], x[1]]])
        else:
            raise Exception("transformation type '{}' not supported".format(transformation_type))

    @staticmethod
    def place_img(src: np.ndarray, dst: np.ndarray, x: int, y: int) -> np.ndarray:
        """Place an image on another image

        Args:
            src (np.ndarray): source image
            dst (np.ndarray): destination image
            x (int): x coordinate of the src image
            y (int): y coordinate of the src image

        Returns:
            np.ndarray: destination image
        """
        h, w = src.shape[:2]
        dst[x : x + h, y : y + w, :] = src
        return dst


def main():
    A = image_stitcher(match_amount=100)
    A.load_imgs(IMG_DIR, reference_frame=1)
    A.stitch([A.HOMOGRAPHY, A.HOMOGRAPHY], display_features=False, display_scale=0.1)
    A.draw(transparency=True, scale=0.3)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
