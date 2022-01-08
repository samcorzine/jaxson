import numpy as np
import cv2

from jaxson import Canvas

# Set up canvas
density = 400
canvas = Canvas(width=density, height=density)

# Drawing logic
sum = canvas.zeros()
sum = sum + canvas.smooth_square(color=[0.5, 0.7, 0.2])

# Rendering
img = np.array(255 * sum)
cv2.imwrite(f"renders/one.png", img)