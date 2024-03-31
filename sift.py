from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST, imread
from functools import cmp_to_key
import logging
import matplotlib.pyplot as plt

EPS = 1e-7
logger = logging.getLogger(__name__)

def generate_base_image(image, sigma, assumed_blur):
    ''' Generating base image from input image. Goal is a upsampling its by 2 in both directions
        and blurring '''
    logger.debug('[DEBUG] Generating base image [generate_base_image]')
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

def calculate_octaves_count(image_shape):
    ''' Calculate a count of octaves in (image pyrimid) '''
    logger.debug('[DEBUG] Calculating count of octaves [calculate_octaves_count]')
    return int(round(log(min(image_shape)) / log(2) - 1))

def generate_gaussian_kernels(sigma, intervals_count):
    ''' Generating a list of gaussian kernels at which to blur the input sigma '''
    logger.debug('[DEBUG] Generating gaussian kernels(scales) [generate_gaussian_kernels]')
    images_per_octave_count = intervals_count + 3
    k = 2 ** (1. / intervals_count)
    gauss_kernels = zeros(images_per_octave_count); gauss_kernels[0] = sigma

    for image_idx in range(1, images_per_octave_count):
        sigma_prev = (k ** (image_idx - 1)) * sigma; sigma_total = k * sigma_prev
        gauss_kernels[image_idx] = sqrt(sigma_total ** 2 - sigma_prev ** 2)
    return gauss_kernels

def generate_gaussian_images(image, count_octaves, gauss_kernels):
    ''' Generating a scale-space pyramid of gaussian images '''
    logger.debug('[DEBUG] Generating gaussian images [generate_gaussian_images]')
    gauss_images = []

    for octave_idx in range(count_octaves):
        gauss_images_in_octave = [image]
        for gauss_kernel in gauss_kernels[1:]:
            gauss_images_in_octave += [GaussianBlur(image, (0, 0), sigmaX=gauss_kernel, sigmaY=gauss_kernel)]
        gauss_images += [gauss_images_in_octave]
        octave_base = gauss_images_in_octave[-3]
        image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)),
                       interpolation=INTER_NEAREST)
    return array(gauss_images, dtype=object)

def generate_DOG_images(gauss_images):
    ''' Generating a Difference-of-Gaussians(DOG) image pyramid '''
    logger.debug('[DEBUG] Generating DOG images [generate_DOG_images]')
    dog_images = []

    for gauss_images_in_octave in gauss_images:
        dog_images_in_octave = []
        for image1, image2 in zip(gauss_images_in_octave, gauss_images_in_octave[1:]):
            dog_images_in_octave += [subtract(image2, image1)]
        dog_images += [dog_images_in_octave]
    return array(dog_images, dtype=object)

def is_pixel_an_extremum(subimg1, subimg2, subimg3, threshold):
    center_pixel_value = subimg2[1, 1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return all(center_pixel_value >= subimg1) and all(center_pixel_value >= subimg3) and \
            all(center_pixel_value >= subimg2[0, :]) and all(center_pixel_value >= subimg2[2, :]) and \
            center_pixel_value >= subimg2[1, 0] and center_pixel_value >= subimg2[1, 2]
        elif center_pixel_value < 0:
            return all(center_pixel_value <= subimg1) and all(center_pixel_value <= subimg3) and \
            all(center_pixel_value <= subimg2[0, :]) and all(center_pixel_value <= subimg2[2, :]) and \
            center_pixel_value <= subimg2[1, 0] and center_pixel_value <= subimg2[1, 2]
    return False

def calculate_gradient_at_center_pixel(pixel_array):
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return array([dx, dy, ds])

def calcualte_hessian_at_center_pixel(pixel_array):
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

def localize_extremum_quadrafit(x, y, image_idx, octave_idx, intervals_count, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, attempts_until_convergence=5):
    extremum_is_outside_image = False; image_shape = dog_images_in_octave[0].shape
    for attempt_idx in range(attempts_until_convergence):
        img1, img2, img3 = dog_images_in_octave[image_idx - 1: image_idx + 2]
        cube = stack([img1[x-1:x+2, y-1:y+2],
                      img2[x-1:x+2, y-1:y+2],
                      img3[x-1:x+2, y-1:y+2]]).astype('float32') / 255
        gradient = calculate_gradient_at_center_pixel(cube)
        hessian = calcualte_hessian_at_center_pixel(cube)
        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < .5 and abs(extremum_update[1]) < .5 and abs(extremum_update[2]) < .5: break

        x += int(round(extremum_update[0]))
        y += int(round(extremum_update[1]))
        image_idx += int(round(extremum_update[2]))
        if x < image_border_width or x >= image_shape[0] - image_border_width or y < image_border_width or y >= image_shape[1] - image_border_width or\
        image_idx < 1 or image_idx > intervals_count:
            extremum_is_outside_image = True; break
    if extremum_is_outside_image:
        return None
    if attempt_idx >= attempts_until_convergence - 1:
        return None
    
    function_value_at_updated_extremum = cube[1,1,1] + .5 * dot(gradient, extremum_update)
    if abs(function_value_at_updated_extremum) * intervals_count >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)

        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            keypoint = KeyPoint()
            keypoint.pt = ( (y + extremum_update[0]) * (2 ** octave_idx), (x + extremum_update[1]) * (2 ** octave_idx) )
            keypoint.octave = octave_idx + image_idx * (2 ** 8) + int(round((extremum_update[2] + .5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ( (image_idx + extremum_update[2]) / float32(intervals_count) )) * (2 ** (octave_idx + 1))
            keypoint.response = abs(function_value_at_updated_extremum)
            return keypoint, image_idx
    return None

def calculate_keypoints_with_orientations(keypoint, octave_idx, gauss_image, radius_factor=3, bins_count=36, peak_ratio=.8, scale_factor=1.5):
    keypoints_with_orientations = []
    image_shape = gauss_image.shape

    scale = scale_factor * keypoint.size / float32(2 ** (octave_idx + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = zeros(bins_count)
    smooth_histogram = zeros(bins_count)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / float32(2 ** octave_idx))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / float32(2 ** octave_idx))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gauss_image[region_y, region_x + 1] - gauss_image[region_y, region_x - 1]
                    dy = gauss_image[region_y - 1, region_x] - gauss_image[region_y + 1, region_x]
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    gradient_orientation = rad2deg(arctan2(dy, dx))
                    weight = exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    histogram_index = int(round(gradient_orientation * bins_count / 360.))
                    raw_histogram[histogram_index % bins_count] += weight * gradient_magnitude

    for n in range(bins_count):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % bins_count]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % bins_count]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            left_value = smooth_histogram[(peak_index - 1) % bins_count]
            right_value = smooth_histogram[(peak_index + 1) % bins_count]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % bins_count
            orientation = 360. - interpolated_peak_index * 360. / bins_count
            if abs(orientation - 360.) < EPS:
                orientation = 0
            new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations

def search_scale_space_extrema(gauss_images, dog_images, intervals_count, sigma, image_border_width, contrast_threshold=.04):
    ''' Search pixels coordinates of all scale-space extrema in the image pyramid '''
    logger.debug('[DEBUG] Searching scale-space extrema [search_scale_space_extrema]')
    threshold = floor(.5 * contrast_threshold / intervals_count * 255)
    keypoints = []

    for octave_idx, dog_images_in_octave in enumerate(dog_images):
        for image_idx, (img1, img2, img3) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (x, y) i s the center of the 3x3 array
            for x in range(image_border_width, img1.shape[0] - image_border_width):
                for y in range(image_border_width, img1.shape[1] - image_border_width):
                    if is_pixel_an_extremum(img1[x-1:x+2, y-1:y+2], img2[x-1:x+2, y-1:y+2], img3[x-1:x+2, y-1:y+2], threshold):
                        local_result = localize_extremum_quadrafit(x, y, image_idx + 1, octave_idx, intervals_count, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                        if local_result is not None:
                            keypoint, localized_image_idx = local_result
                            keypoints_with_orientations = calculate_keypoints_with_orientations(keypoint, octave_idx, gauss_images[octave_idx][localized_image_idx])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints += [keypoint_with_orientation]
    return keypoints

def compare_keypoints(kp1, kp2):
    if kp1.pt[0] != kp2.pt[0]:
        return kp1.pt[0] - kp2.pt[0]
    if kp1.pt[1] != kp2.pt[1]:
        return kp1.pt[1] - kp2.pt[1]
    if kp1.size != kp2.size:
        return kp1.size - kp1.size
    if kp1.angle != kp2.angle:
        return kp1.angle - kp2.angle
    if kp1.response != kp2.response:
        return kp1.response - kp1.response
    if kp1.octave != kp2.octave:
        return kp2.octave - kp1.octave
    return kp2.class_id - kp1.class_id

def remove_dublicate_keypoints(keypoints):
    """Sort keypoints and remove duplicate keypoints
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compare_keypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

def convert_keypoints_to_input_image_size(keypoints):
    """Convert keypoint point, size, and octave to input image size
    """
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints

def unpack_octave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
    return octave, layer, scale

def generate_descriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """Generate descriptors for each keypoint
    """
    logger.debug('Generating descriptors...')
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpack_octave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = round(scale * array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = cos(deg2rad(angle))
        sin_angle = sin(deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(half_width, sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = sqrt(dx * dx + dy * dy)
                        gradient_orientation = rad2deg(arctan2(dy, dx)) % 360
                        weight = exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            # Smoothing via trilinear interpolation
            # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
            # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
            row_bin_floor, col_bin_floor, orientation_bin_floor = floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(norm(descriptor_vector), EPS)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return array(descriptors, dtype='float32')

def calculate_keypoints_N_desctiptions(image, sigma=1.6, intervals_count=3, assumed_blur=.5, image_border_width=5):
    ''' Calculate SIFT keypoints and descriptions for an input image '''
    logger.debug('[DEBUG] SIFT calculating started [calculate_keypoints_N_desctiptions]')

    image = image.astype('float32')
    base_image = generate_base_image(image, sigma, assumed_blur)
    octaves_count = calculate_octaves_count(base_image.shape)
    gauss_kernels = generate_gaussian_kernels(sigma, intervals_count)
    gauss_images = generate_gaussian_images(base_image, octaves_count, gauss_kernels)
    dog_images = generate_DOG_images(gauss_images)
    keypoints = search_scale_space_extrema(gauss_images, dog_images, intervals_count, sigma, image_border_width)
    keypoints = remove_dublicate_keypoints(keypoints)
    keypoints = convert_keypoints_to_input_image_size(keypoints)
    descriptions = generate_descriptors(keypoints, gauss_images)
    return keypoints, descriptions