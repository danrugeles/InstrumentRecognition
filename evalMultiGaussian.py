from numpy import *
def evalMultiGaussian(mean,cov,point):
	dimension = len(point.shape)
	detCov = sqrt(linalg.det(cov))
	frac = (2*pi)**(-dimension/2.0) * (1/detCov)
	fprime = (mean - point)**2
	return frac * exp(-0.5*dot(dot(array(fprime), linalg.inv(cov)),array(fprime).T))


if __name__ == '__main__':

	mean=array([2,3])
	cov=array([[2,0],[0,5]])
	print evalMultiGaussian(mean,cov,array([2,3]))
	print evalMultiGaussian(mean,cov,array([0,0]))
	print evalMultiGaussian(mean,cov,array([5,5]))

