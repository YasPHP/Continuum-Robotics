# OpenCV [Core Operations](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_table_of_contents_core/py_table_of_contents_core.html)
## May 11th, 2021.

# Loading and displaying an image
- image loading (imread): gets image from path
- image showing (imshow): prints image- varying function syntax from google collab [cv_imshow vs cv.imshow]
- making image greyscale (set final param in imread to 0)
- image saving (imwrite)
- image pixel access
- image pixel editing (img.item and img.itemset)

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

