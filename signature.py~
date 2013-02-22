from cv2 import *
from numpy import *
from mysys import *
import pylab 
from signals import *

DEBUG=False


#-- SignatureContour ---------------------------------
#
# Given a contour a, returns the signature of the object
#
#-------------------------------------------------
def findSignature(a):

	M = moments(a)
	center = (array([M['m10'],M['m01']])/M['m00']).astype(int)	


	a=reshape(a,(a.shape[0],2))
	if DEBUG:	print center,"center"

	#**Find all possible candidates to start the string
	init=(a[:][:,1]<center[1]+5) & (a[:][:,1]>center[1]-5)		
	if DEBUG:	print a[init],"candidates"

	#**Select the candidate tp the left most side
	candidateidx=array(range(0,len(init)))

	mini=inf
	minidx=0
	if DEBUG:	print candidateidx[init],"candidatesidx"

	for candidate in candidateidx[init]:
		if a[candidate][0]<mini:
			mini=a[candidate][0]
			minidx=candidate

	#**Start point is the best candidate (a[minidx])
	if DEBUG: 	print a[minidx],"best cand"

	#**Rearreange the order of contours
	signature=vstack((a[minidx:],a[:minidx]))

	if DEBUG:	print signature[:10],"signature first samples"

	return ((signature[:][:,0]-center[0])**2+(signature[:][:,1]-center[1])**2)**0.5

if __name__ == '__main__':

	#Specific cases
	a=load("storage/cnt20_18.npy")
	b=load("storage/cnt5_8.npy")

	res=matchShapes(a,b,cv.CV_CONTOURS_MATCH_I3,0)
	print res,"res"

	signal1=findSignature(a)
	signal2=findSignature(b)
	
	print ncc2(signal1,signal2),"ncc"


	xxx=100*zeros((3000,3000))
	drawContours(xxx,[b],0,255,-1)
	drawContours(xxx,[a],0,100,-1)
	imwrite("Case.jpg",xxx)
	


	pylab.plot(signal1) 
	pylab.plot(signal2) 
	
	pylab.xlabel('signature')
	pylab.ylabel('L2 Norm')
	pylab.title('Signature of contour')
	pylab.grid(True)
	pylab.savefig('signature.jpg')
	#pylab.show()


	#print a[init]
	#print a[init].argmin(axis=0)
	#print "hell"
	#print a[init].min(axis=0)

	#xxx=zeros((1000,1000))
	#for  point in a:
	#	circle(xxx,(point[0],point[1]), 1,(100,100,100),1)
	#imwrite("exiit.jpg",xxx)

