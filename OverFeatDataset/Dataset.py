import abc


class Dataset(object):
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def numclasses(self):
		""" Return the number of classes in the dataset"""
		return


	@abc.abstractmethod
	def dbname(self):
		""" Return the name of the dataset"""
		return


	@abc.abstractmethod
	def ispresent(self, category):
		""" Return True if the category is present in the dataset, otherwise false"""
		return


	@abc.abstractmethod
	def __predefinedsplit(self):
		""" Return True if the training and testing sets are predefined by the dataset provider, otherwise False"""
		return


	@abc.abstractmethod
	def getdata(self, categories, train_share=0.6, test_share=0.2, val_share=0.2,
				forcesplit=False):
		""" Return 3 lists - one each for training, testing and validation. Also return a dictionary mapping
		the categorynames to label numbers. For negativecategories='others', all categories other than the positive
		category are taken as the negative class"""
		return


