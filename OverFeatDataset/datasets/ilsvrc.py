import sys
import os
import inspect
scriptfolder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.join(scriptfolder,'..','..'	))
from OverFeatDataset import Dataset
import re
import pandas as pd
import xml.etree.ElementTree as ET
import random
from datetime import datetime
import numpy as np


def merge_dict_nooverwrite(dict1, dict2):
	merged = {}
	for k in dict1.keys():
		if k not in merged:
			merged[k] = []
		merged[k] = merged[k] + dict1[k]

	for k in dict2.keys():
		if k not in merged:
			merged[k] = []
		merged[k] = merged[k] + dict2[k]

	return merged

class ILSVRC(object):


	def __init__(self, wordnetmappingfilename, base_save_path):
		assert base_save_path is not None, "You must provide the base path where ILSVRC2015 is stored."
		assert wordnetmappingfilename is not None, "You must provide the path where the wordnet mapping file is stored."
		assert os.path.isdir(base_save_path), "The base path for ILSVRC2015 is either not a path or it does not exist."
		self.base_save_path = base_save_path
		self.mapping = dict()
		self.inversemapping = dict()
		try:
			file = open(wordnetmappingfilename, 'r')
		except:
			raise IOError('The file was not found or could not be read.')

		for line in file:
			line = re.split('[,\t\n]',line.strip())
			line = [(x.lower().strip()) for x in line]
			self.mapping[line[0]] = line[1:]
			for category in line[1:]:
				if category in self.inversemapping:
					self.inversemapping[category].append(line[0])
				else:
					self.inversemapping[category] = [line[0]]
		file.close()


	def numclasses(self):
		return 1000


	def dbname(self):
		return "ILSVRC2015"


	def __ispresent(self, category):
		assert category is not None, "category must be a string representing a class you want to query."
		assert type(category) is str, "category must be a string representing a class you want to query."
		try:
			wnid = self.inversemapping[category.lower()]
		except KeyError:
			return False
		return True


	def __getwnid(self, category):
		try:
			wnid = self.inversemapping[category.lower()]
		except KeyError:
			return None
		return wnid

	def __getcategorynames(self, wnid):
		try:
			categories = self.mapping[wnid.lower()]
		except KeyError:
			return None
		return categories



	def __predefinedsplit(self):
		return True


	def __readtrain(self):
		imagesetfile = os.path.join(self.base_save_path,'ImageSets','CLS-LOC','train_cls.txt')
		trainingsourcepath = os.path.join(self.base_save_path,'Data','CLS-LOC','train')
		df = pd.read_table(imagesetfile, header=None, delimiter=' ')
		df = df.ix[:,0]
		categories = set()
		filenames = dict()
		for index in range(len(df)):
			classsearch = re.search('^(n[0-9]+)/(.+)$', df[index])
			categories.add(classsearch.group(1))
			if classsearch.group(1) in filenames:
				filenames[classsearch.group(1)].append(os.path.join(trainingsourcepath,classsearch.group(2) + '.JPEG'))
			else:
				filenames[classsearch.group(1)] = [os.path.join(trainingsourcepath,classsearch.group(2) + '.JPEG')]

		categories = list(categories)
		self.categorymapping = dict()
		for key, value in zip(categories, range(1, len(categories) + 1)):
			self.categorymapping[key] = value

		return filenames


	def __readvalidation(self):
		imagesetfile = os.path.join(self.base_save_path,'ImageSets','CLS-LOC','val.txt')
		validationsourcepath = os.path.join(self.base_save_path,'Data','CLS-LOC','val')
		validationannotationpath = os.path.join(self.base_save_path,'Annotations','CLS-LOC','val')
		df = pd.read_table(imagesetfile, header=None, delimiter=' ')
		df = df.ix[:,0]
		filenames = dict()
		for i in range(len(df)):
			category = self.__readvoc(os.path.join(validationannotationpath, df[i] + '.xml'))
			if category in filenames:
				filenames[category].append(os.path.join(validationsourcepath, df[i] + '.JPEG'))
			else:
				filenames[category] = [os.path.join(validationsourcepath, df[i] + '.JPEG')]
		return filenames


	def __readtest(self):
		imagesetfile = os.path.join(self.base_save_path, 'ImageSets', 'CLS-LOC', 'test.txt')
		testsourcepath = os.path.join(self.base_save_path, 'Data', 'CLS-LOC', 'test')
		df = pd.read_table(imagesetfile, header=None, delimiter=' ')
		df = df.ix[:,0]
		df = df.tolist()
		print df
		df = map(lambda x: os.path.join(testsourcepath, x + '.JPEG'), df)
		return df

	def __readvoc(self, filename):
		file = ET.parse(filename)
		r = file.getroot()
		for i in r.getchildren():
			if i.tag == 'object':
				for j in i.getchildren():
					if j.tag == 'name':
						return j.text


	def getdata(self, categories, train_share=None, test_share=None, val_share=None,
				forcesplit=False):

		for y in [x for x in categories if x != 'others']:
			if not self.__ispresent(y):
				assert False, """You have specified a category that is not present in ILSVRC. You must provide cate
				gories which only exist in ILSVRC."""

		trainingdata = self.__readtrain()
		#testingdata = self.__readtest()
		validationdata = self.__readvalidation()

		hasothers = any(x == 'others' for x in categories)

		if not hasothers:
			trainingdata = dict((k,r) for k,r in trainingdata.iteritems() if k in categories)
			validationdata = dict((k,r) for k,r in validationdata.iteritems() if k in categories)
		else:
			trainingdata['others'] = []
			validationdata['others'] = []
			for k in trainingdata.keys():
				if k == 'others':
					continue
				if not k in categories:
					trainingdata['others'] = trainingdata['others'] + trainingdata[k]
					trainingdata.pop(k, None)

			for k in validationdata.keys():
				if k == 'others':
					continue
				if not k in categories:
					validationdata['others'] = validationdata['others'] + validationdata[k]
					validationdata.pop(k, None)

		if forcesplit:
			trainingdata = merge_dict_nooverwrite(trainingdata, validationdata)
			validationdata = {}
			for k in trainingdata.keys():
				total_num = len(trainingdata[k])
				num_train = np.ceil(train_share * total_num)
				num_val = total_num - num_train
				assert num_val > 0,"""There must be at least one training data point for each category."""
				assert num_val > 0,"""There must be at least one validation data point for each category."""
				random.seed(datetime)
				val_ind = random.sample(range(total_num), num_val)
				validationdata[k] = [trainingdata[k][x] for x in val_ind]
				for i in sorted(val_ind, reverse=True):
					del trainingdata[i]

		labels = [ self.__getcategorynames(x) for x in trainingdata.keys()]
		labelmapping  = dict(zip(labels,range(1,len(labels)+1)))
		return trainingdata, validationdata, labelmapping


Dataset.Dataset.register(ILSVRC)













