import numpy as np
import cv2


def detect_text_bounds(image: np.array) -> (int, int):
    """
    Find the lower and upper bounding lines in an image of a word
    """
    if len(image.shape) >= 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) >= 3 and image.shape[2] == 1:
        image = np.squeeze(image, axis=-1)

    _, threshold = cv2.threshold(image, 0.8, 1, cv2.THRESH_BINARY_INV)

    line_sums = np.sum(threshold, axis=1).astype(float)
    line_sums = np.convolve(line_sums, np.ones(5) / 5, mode='same')

    line_sums_d = np.diff(line_sums)

    std_factor = 0.5
    min_threshold = np.mean(line_sums_d[line_sums_d <= 0]) - std_factor * np.std(line_sums_d[line_sums_d <= 0])
    bottom_index = np.max(np.where(line_sums_d < min_threshold))

    max_threshold = np.mean(line_sums_d[line_sums_d >= 0]) + std_factor * np.std(line_sums_d[line_sums_d >= 0])
    top_index = np.min(np.where(line_sums_d > max_threshold))

    return bottom_index, top_index


def dist(p_one, p_two) -> float:
    return np.linalg.norm(p_two - p_one)


def crop(image: np.array, ratio: float = None, pixels: int = None) -> np.array:
    assert ratio is not None or pixels is not None, "Please specify either pixels or a ratio to crop"

    width, height = image.shape[:2]

    if ratio is not None:

        width_crop = int(ratio * width)
        height_crop = int(ratio * height)
    else:
        width_crop= pixels
        height_crop = pixels

    return image[height_crop:height-height_crop, width_crop:width-width_crop]


def find_target_points(top_left, top_right, bottom_left, bottom_right):
    max_width = max(int(dist(bottom_right, bottom_left)), int(dist(top_right, top_left)))
    max_height = max(int(dist(top_right, bottom_right)), int(dist(top_left, bottom_left)))
    destination_corners = [[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]]

    return order_points(destination_corners)


def order_points(points: np.array) -> tuple:
    """
    inspired by: https://learnopencv.com/automatic-document-scanner-using-opencv/
    """
    sum = np.sum(points, axis=1)
    top_left = points[np.argmin(sum)]
    bottom_right = points[np.argmax(sum)]

    diff = np.diff(points, axis=1)
    top_right = points[np.argmin(diff)]
    bottom_left = points[np.argmax(diff)]

    return top_left, top_right, bottom_left, bottom_right


def get_page(image: np.array) -> np.array:
    """
    inspired by: https://github.com/Kakaranish/OpenCV-paper-detection
    """
    filtered = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.medianBlur(filtered, 11)

    canny = cv2.Canny(filtered, 30, 50, 3)
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    max_perimeter = 0
    max_contour = None
    for contour in contours:
        contour = np.array(contour)
        perimeter = cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if perimeter > max_perimeter and cv2.isContourConvex(contour_approx) and len(contour_approx) == 4:
            max_perimeter = perimeter
            max_contour = contour_approx

    if max_contour is not None:
        max_contour = np.squeeze(max_contour)
        points = order_points(max_contour)

        target_points = find_target_points(*points)
        M = cv2.getPerspectiveTransform(np.float32(points), np.float32(target_points))
        final = cv2.warpPerspective(image, M, (target_points[3][0], target_points[3][1]), flags=cv2.INTER_LINEAR)
        final = crop(final, pixels=10)
        return final

    return image


def get_words(page: np.array, dilation_size: int = 3):
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 125, 1, cv2.THRESH_BINARY_INV)

    dilation_size = dilation_size
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                       (dilation_size, dilation_size))
    thresholded = cv2.dilate(thresholded, element, iterations=3)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    words = []
    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        ratio = w / h
        if ratio <= 0.1 or ratio >= 10.0:
            continue
        boxes.append([x, y, w, h])
        words.append(page[y:y+h, x:x+w])

    return words, boxes