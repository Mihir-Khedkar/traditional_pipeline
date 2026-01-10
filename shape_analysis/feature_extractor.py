import cv2
import numpy as np
from typing import Optional, Tuple


class HoughLineDetector:

    def __init__(self, canny_edges: np.ndarray):
        if canny_edges.ndim != 2:
            raise ValueError("Canny image must be a single-channel (grayscale) image.")

        self.edges = canny_edges.copy()

    def detect_standard(self, rho: float = 1.0, theta: float = np.pi / 180, threshold: int = 150) -> Optional[np.ndarray]:
        lines = cv2.HoughLines(self.edges, rho=rho, theta=theta, threshold=threshold)
        return lines

    def detect_probabilistic(self, rho: float = 1.0, theta: float = np.pi / 180, threshold: int = 100, min_line_length: int = 50, max_line_gap: int = 10) -> Optional[np.ndarray]:
        lines = cv2.HoughLinesP(self.edges, rho=rho, theta=theta, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        return lines

    @staticmethod
    def draw_standard(image: np.ndarray, lines: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
        output = image.copy()

        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(output, (x1, y1), (x2, y2), color, thickness)
        return output

    @staticmethod
    def draw_probabilistic(image: np.ndarray, lines: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        output = image.copy()

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), color, thickness)

        return output
