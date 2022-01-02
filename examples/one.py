import time
import numpy as np
import cv2

from jaxson import Canvas

# Set up canvas
density = 400
canvas = Canvas(shape=(1, density, density))

# Drawing logic
sum = canvas.zeros()
sum = sum + canvas.circle()

# Rendering
img = np.array(255 * sum)
cv2.imwrite(f"renders/one.png", img)
