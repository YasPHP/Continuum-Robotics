# OpenCV [Core Operations](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_table_of_contents_core/py_table_of_contents_core.html)
## May 11th, 2021.

# Loading and displaying an image


# Basic Operations on images
- image properties (shape, size, dtype)
- image ROI (region of interest)
- splitting and merging image channels
- padding (making borders for images)

# Arithmetic Operations on images
- image addition: two types include OpenCV addition (preferable b/c saturated operations) and Numpy addition (modulo operation)
- when blending two images, the shape of both must be the same (or resized otherwise) when using:
<code> dst = cv2.addWeighted(img1,0.5,resized_img2,0.3,0) </code>
- bitwise operations

# Performance Measurement and Improvement Techniques

