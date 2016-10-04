import cv2
import random
from datetime import datetime

def resize_image(im, resize_val, resize_type):
	chkresize_type = sum([resize_type == 'fixed',
						  resize_type == 'min',
						  resize_type == 'max'])

	assert chkresize_type == 1, "resize_type must be one of fixed, min or max"

	if resize_type == 'fixed':
		return resize_fixed(im, resize_val)

	if resize_type == 'min':
		return resize_min(im, resize_val)

	if resize_type == 'max':
		return resize_max(im, resize_val)


def resize_fixed(im, resize_val):
	chkresize_val = sum([type(resize_val) is int and resize_val > 0,
						 type(resize_val) is tuple and len(resize_val) ==2 and all(isinstance(x, int) for x in
																				   resize_val) and
						 all(x>0 for x in resize_val)])

	assert chkresize_val == 1, "resize_val must either be a positive integer or a tuple of two positive integers"

	if type(resize_val) is int:
		resize_val = (resize_val, resize_val)

	im = cv2.resize(im, resize_val, interpolation=cv2.INTER_CUBIC)

	return im


def resize_min(im, resize_val):
	assert type(resize_val) == int, "resize_val must be a positive integer"

	sz = im.shape
	sz = tuple(float(x) for x in sz)

	aspect_ratio = sz[0]/sz[1]

	if sz[0] < sz[1]:
		new_height = int(resize_val)
		new_width = int(resize_val/aspect_ratio)
	else:
		new_height = int(resize_val * aspect_ratio)
		new_width = int(resize_val)

	im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

	return im


def resize_max(im, resize_val):
	assert type(resize_val) == int, "resize_val must be a positive integer"

	sz = im.shape
	sz = tuple(float(x) for x in sz)

	aspect_ratio = sz[0] / sz[1]

	if sz[0] < sz[1]:
		new_height = int(resize_val * aspect_ratio)
		new_width = int(resize_val)

	else:
		new_height = int(resize_val)
		new_width = int(resize_val / aspect_ratio)

	im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

	return im


def flip_image(im):
	im = cv2.flip(im, 1)
	return im


def random_crop(im, crop_size):
	sz = im.shape
	chkcrop_size = all([type(crop_size) is tuple and len(crop_size) == 2 and all(isinstance(x,int) for x in crop_size)
						and all(x>0 for x in crop_size),
						crop_size[0] <= sz[0] and crop_size[1] <= sz[1]])

	assert chkcrop_size, "crop_size must be a tuple of two positive integers within the image dimensions"

	comp_tuple = all(x==y for x,y in zip(crop_size,sz))
	if comp_tuple:
		return im

	row_lim = sz[0] - crop_size[0]
	col_lim = sz[1] - crop_size[1]

	random.seed(datetime.now())
	row_lim = random.randint(0, row_lim)
	col_lim = random.randint(0, col_lim)

	im = im[row_lim : row_lim + crop_size[0], col_lim : col_lim + crop_size[1], :]
	return im

