import cv2
import numpy as np
import matplotlib.pyplot as plt

color = np.random.randint(0, 255, (200, 3))
fourcc = cv2.VideoWriter_fourcc(*"XVID")


def get_affine_parameters(M: np.ndarray) -> np.ndarray:
    """Get tx, ty, theta and a from the matrix M of an affime transform

    Args:
        M (np.ndarray): transformation matrix

    Returns:
        np.ndarray: [tx, ty, theta, a]
    """
    tx = M[0, 2]
    ty = M[1, 2]
    theta = np.arctan2(M[1, 0], M[0, 0])
    a = np.sqrt(np.linalg.det(M[0:2, 0:2]))
    return np.array([tx, ty, theta, a])


def compute_matched_points(src_img, dst_img, match_amount: int = 100):
    """Compute matched keypoints between two images

    Args:
        src_img: source image
        dst_img: destination image
        match_amount (int, optional): max number of matched keypoints desired. Defaults to 100.

    Returns:
        pts1, pts2: list of matched keypoints for both images
    """
    kp_src, desc_src = compute_features_orb(src_img)
    kp_dst, desc_dst = compute_features_orb(dst_img)
    pts1, pts2 = match_points_bf(kp_src, kp_dst, desc_src, desc_dst, match_amount=match_amount)
    return pts1, pts2


def compute_features_orb(img):
    """Computes features of an image using ORB

    Args:
        img: input image

    Returns:
        kp, desc: key points and descriptors of the features
    """
    orb = cv2.ORB_create()
    kp, desc = orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None)
    return kp, desc


def match_points_bf(kp_src, kp_dst, desc_src, desc_dst, match_amount: int):
    """Matches two different sets of keypoints using their descriptors

    Args:
        kp_src: source keypoints
        kp_dst: destination keypoints
        desc_src: descriptors of source keypoints
        desc_dst: descriptors of destination keypoints
        match_amount (int): max number of match desired

    Returns:
        src_pts, dst_pts: list of matched keypoints for both src and dst
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_src, desc_dst)
    matches = sorted(matches, key=lambda x: x.distance)[:match_amount]
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    return src_pts, dst_pts


def moving_average(curve: np.ndarray, window: int) -> np.ndarray:
    """Apply moving average to an inputed curve

    Args:
        curve (np.ndarray): curve to average
        window (int): window of the moving average

    Returns:
        np.ndarray: averaged curve
    """
    window_size = 2 * window + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.pad(curve, (window, window), mode="edge")
    curve_smoothed = np.convolve(curve_pad, f, mode="same")
    curve_smoothed = curve_smoothed[window:-window]
    return curve_smoothed


def smooth_trajectory(trajectory: np.ndarray, window: int = 30) -> np.ndarray:
    """Apply moving average to all four parameters (x, y, theta and a) curves

    Args:
        trajectory (np.ndarray): trajectory (x, y, theta, a)
        window (int, optional): window of the moving average. Defaults to 30.

    Returns:
        np.ndarray: averaged trajectory
    """
    smoothed_trajectory = np.zeros(trajectory.shape)
    for k in range(4):
        smoothed_trajectory[:, k] = moving_average(trajectory[:, k], window=window)
    return smoothed_trajectory


def zoom_in(img, factor: float):
    """Zoom in on an image to a specified amount. Outputed image has the same dimension as the input image

    Args:
        img: input image
        factor (float): zoom factor. Must be greater or equal than 1

    Raises:
        Exception: when factor is less than 1

    Returns:
        new_img: zoomed in image
    """
    if factor < 1:
        raise Exception("Factor < 1 not supported")
    h, w = img.shape[:2]
    new_img = cv2.resize(img, None, fx=factor, fy=factor)
    new_h, new_w = new_img.shape[:2]
    x, y = (new_h - h) // 2, (new_w - w) // 2
    return new_img[x : x + h, y : y + w, :]


def compute_video_points(
    filename: str, match_amount: int = 100, display: bool = False, display_scale: float = 0.4, save_file: str = ""
):
    """Get a list of matched keypoints between each frame of a video

    Args:
        filename (str): path to video
        match_amount (int, optional): max amount of keypoints desired. Defaults to 100.
        display (bool, optional): wether to display keypoints. Defaults to False.
        display_scale (float, optional): scale of the displayed image. Defaults to 0.4.
        save_file (str, optional): file to save the video with points, does not save if equal to "". Defaults to "".

    Raises:
        Exception: video fails to be read

    Returns:
        list: list of tupple of list containing the keypoints for each frame change (pts1, pts2)
    """
    cap = cv2.VideoCapture(filename)
    ret, old_frame = cap.read()
    if not ret:
        cap.release()
        raise Exception("Failed to read video")
    mask = np.zeros_like(old_frame)
    points = []
    if save_file:
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(save_file, fourcc, fps, (old_frame.shape[1], old_frame.shape[0]))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pts1, pts2 = compute_matched_points(old_frame, frame, match_amount=match_amount)
        points.append((pts1, pts2))
        old_frame = frame.copy()
        if display:
            for i, (new, old) in enumerate(zip(pts1, pts2)):
                a, b = new
                c, d = old
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)
            cv2.imshow("Optical Flow", cv2.resize(img, None, fx=display_scale, fy=display_scale))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if save_file:
                out.write(img)
    if save_file:
        out.release()

    return points


def compute_trajectory(points) -> np.ndarray:
    """Compute the trajectory (x, y, theta, a) corresponding to the parameter of an affine transformation from a list of matched points

    Args:
        points (list): list containing for each frame change the lists of matched keypoints

    Returns:
        np.ndarray: trajectory (x, y, thete, a) corresponding to the transformation between the points
    """
    trajectory = np.zeros((len(points) + 1, 4), dtype=np.float32)
    trajectory[0] = np.array([0, 0, 0, 1])
    for i, (pts1, pts2) in enumerate(points):
        M = cv2.estimateAffinePartial2D(pts1, pts2)[0]
        if M is None:
            M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        p = get_affine_parameters(M)
        trajectory[i + 1, :3] = trajectory[i, :3] + p[:3]
        # neutral value for a is 1, therefore we multiply instead of adding
        trajectory[i + 1, 3] = p[3] * trajectory[i, 3]
    return trajectory


def plot_trajectory(trajectory: np.ndarray, smoothed_trajectory: np.ndarray):
    """Plots trajectory and smoothed trajectory for each variable (x, y, theta,a )

    Args:
        trajectory (np.ndarray): trajectory
        smoothed_trajectory (np.ndarray): smoothed trajectory
    """
    _, axs = plt.subplots(4, 1)
    names = ["x", "y", "theta", "a"]
    for k in range(4):
        plt.plot()
        axs[k].plot(trajectory[:, k])
        axs[k].plot(smoothed_trajectory[:, k])
        axs[k].set_title(names[k])
        axs[k].legend([names[k], names[k] + "_smoothed"])
    plt.show()


def compute_corrective_transforms(trajectory, smoothed_trajectory):
    """Compute corrective transforms to apply to the images

    Args:
        trajectory (_type_): trajectory (x, y, theta, a)
        smoothed_trajectory (_type_): smoothed trajectory (x, y ,theta , a)

    Returns:
        list: list of corrective affine transformation matrixes
    """
    transforms = []
    for i in range(len(trajectory[:, 0])):
        dx = smoothed_trajectory[i][0] - trajectory[i][0]
        dy = smoothed_trajectory[i][1] - trajectory[i][1]
        dtheta = smoothed_trajectory[i][2] - trajectory[i][2]
        # Since a is has a multiplicative impact, we divide to get the corrective amount
        da = smoothed_trajectory[i][3] / trajectory[i][3]
        M = np.array(
            [
                [da * np.cos(dtheta), -da * np.sin(dtheta), dx],
                [da * np.sin(dtheta), da * np.cos(dtheta), dy],
            ],
            dtype=np.float32,
        )
        transforms.append(M)
    return transforms


def apply_transforms(transforms: list, video_in: str, video_out: str, zoom_in_factor: float = 1):
    """Apply the transformation matrixes to their corresponding frame on the inputed video and zoom in to specified amount

    Args:
        transforms (list): list of affine transformation matrixes
        video_in (str): path of input video
        video_out (str): path of output video
        zoom_in_factor (float, optional): amount of zoom on the frames. Defaults to 1.
    """
    cap = cv2.VideoCapture(video_in)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_out, fourcc, 25.0, (width, height))
    for transform in transforms:
        ret, frame = cap.read()
        if not ret:
            break
        stabilized_frame = cv2.warpAffine(frame, transform, (width, height))
        out.write(zoom_in(stabilized_frame, zoom_in_factor))

    cap.release()
    out.release()


def stabilize_video(video_in: str, video_out: str, display: bool = False):
    """Stabilize the input video

    Args:
        video_in (str): path of input video
        video_out (str): path of output video
        display (bool, optional): wether to display intermediate results. Defaults to False.
    """
    print("Matching points")
    points = compute_video_points(video_in, match_amount=100, display=display, save_file="optical_flow.mp4")
    print("Computing trajectory from points")
    trajectory = compute_trajectory(points)
    print("Smoothing trajectory")
    smoothed_trajectory = smooth_trajectory(trajectory, window=30)
    print("Computing corrective transforms")
    transforms = compute_corrective_transforms(trajectory, smoothed_trajectory)
    if display:
        plot_trajectory(trajectory, smoothed_trajectory)
    print("Applying corrective transforms")
    apply_transforms(transforms, video_in, video_out, zoom_in_factor=1.1)


if __name__ == "__main__":
    stabilize_video("C0004.MP4", "stabilized_video.mp4", display=True)
