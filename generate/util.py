import numpy as np


def stack_lines(lines: list, h_gap: int = 6):
    width = max([im.shape[1] for im in lines])
    height = (lines[0].shape[0] + h_gap) * len(lines)

    result = np.ones((height, width)) * 255

    y_pos = 0
    for line in lines:
        result[y_pos:y_pos + line.shape[0], 0:line.shape[1]] = line
        y_pos += line.shape[0] + h_gap

    return result