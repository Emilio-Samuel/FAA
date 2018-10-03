from abc import ABCMeta,abstractmethod
class EstrategiaParticionado (object):
	__metaclass__ = ABCMeta
	def __init__(self,nombreEstrategia, numeroParticiones,datos):
		self.nombreEstrategia = nombreEstrategia
		self.numeroParticiones = numeroParticiones
		self.particiones = list()
		self.datos = datos
	@abstractmethod
	def creaParticiones(self):
		pass