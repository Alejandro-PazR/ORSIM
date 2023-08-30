class SpaceCraft:
	def __init__(self,
	    	mass: float,
			CD: float,
			A_drag: float,
			CR: float,
			A_srp: float,
		):
		self._mass = mass
		self._CD = CD
		self._A_drag = A_drag
		self._CR = CR
		self._A_srp = A_srp

	@property
	def mass(self):
		return self._mass

	@property
	def CD(self):
		return self._CD

	@property
	def A_drag(self):
		return self._A_drag
	
	@property
	def CR(self):
		return self._CR
	
	@property
	def A_srp(self):
		return self._A_srp