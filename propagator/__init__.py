class BasePropagator:
	def __init__(self,
			spacecraft,
			state,
			epoch,
			tStop,
		):
		self._spacecraft = spacecraft
		self._state = state
		self._epoch = epoch
		self._tStop = tStop