from numpy import *

WARNING=False

#-- Decrease to size ---------------------------------
#
#  Decrease the size of an array by "discretization"
#
#-------------------------------------------------
def decreaseToSize(b,size):

	newidxs=rint(arange(0,len(b),len(b)/float(size))).astype(int)

	if len(newidxs)>size:
		if WARNING: print "warning: size went up to"+str(len(newidxs))+"and it should have been"+str(size)
		return  b[newidxs[:size]]
	else:
 		return  b[newidxs]




#-- NCC ---------------------------------
#
# Between two signals of the DIFFERENT length
# The longer length is adjusted to the smaller length
#
#-------------------------------------------------
def ncc2 (a,b):
	if(len (a)== len(b)):
		return ncc(a,b)
	elif len(a)>len(b):
		newa=decreaseToSize(a,len(b))			
		return ncc(newa,b)
	else:
		newb=decreaseToSize(b,len(a))	
		return ncc(newb,a)

#-- NCC ---------------------------------
#
# Between two signals of the same length
#
#-------------------------------------------------
def ncc(a,b):
	#print "cc",((a-a.mean())*(b-b.mean())).sum()
	return ((a-a.mean())*(b-b.mean())).sum()/(((a-a.mean())**2).sum()*((b-b.mean())**2).sum())**0.5

if __name__=="__main__":
	a=array([2,4,2,4,2])
	b=array([1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2])
	
	print ncc2(a,-a)
	print ncc2(b,b)
	print ncc2(b,a)
	print ncc2(a,b)




