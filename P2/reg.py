def reg_log_v1(nu, nepocas, ejemplos,x):
	omega = ones((1,len(x)))
	a = sum(omega.*x)
	for(ie=0;ie<nepocas;ie++):
		for(i=0;i<ejemplos;i++):
			omega = omega-nu*(1/1+np.exp())
	return omega
