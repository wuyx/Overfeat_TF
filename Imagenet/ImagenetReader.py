import re

class ImagenetReader:
	def __init__(self, wordnetmappingfilename):
		self.mapping = dict()
		for line in open(wordnetmappingfilename,'r'):
			line = re.split('[,\t\n]',line.strip())
			line = [(x.lower().strip()) for x in line]
			for category in line[1:]:
				if category in self.mapping:
					self.mapping[category].append(line[0])
				else:
					self.mapping[category] = [line[0]]

	def wnidfromclass(self, classname):
		try:
			wnid = self.mapping[classname.lower()]
			return wnid
		except KeyError:
			print('{} was not found in the mapping file.'.format(classname.lower()))



