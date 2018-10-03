import random as r
from EstrategiaParticionado import EstrategiaParticionado
from Particion import Particion
from abc import ABCMeta,abstractmethod
import numpy as np
class ValidacionSimple (EstrategiaParticionado):

	def __init__(self,numeroParticiones, porcentaje,datos):
		super().__init__("Validacion simple",numeroParticiones,datos)
		self.porcentaje = porcentaje
	@abstractmethod
	def creaParticiones(self):
		ntot = len(self.datos.datos)
		n = int(ntot*self.porcentaje)
		for i in range(self.numeroParticiones):
			x = list(range(ntot))
			r.shuffle(x)
			p = Particion(x[:n],x[(n+1):])
			self.particiones.append(p)
