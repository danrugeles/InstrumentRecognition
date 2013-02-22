from cv2 import *
from numpy import *
import operator
import itertools
import time
from scipy.spatial.distance import mahalanobis
import sys
from mysys import *
from myblobs import *
from normalization import *
from signature import *


MINSIZECONTOUR=200
SEGMENTATIONPROB=0

MORPHOLOGYCLOSINGS=3
Toy=False
PROCESS=True
DEBUG=False
BLOBNOISE=20
MINBLOBSIZE=100

INTERACTIVE=False


#-- Normalize ---------------------------------
#
# Deletes the pixels inside the specified contour
#
#-------------------------------------------------
def normalize(imghls):
	normalization=((imghls[:][:,:][:,:,0]+1.0)**2+(imghls[:][:,:][:,:,1]+1.0)**2+(imghls[:][:,:][:,:,2]+1.0)**2)**0.5
	normalization=array([normalization])
	imghls[:][:,:][:,:,0]=255*(imghls[:][:,:][:,:,0]/normalization)
	imghls[:][:,:][:,:,1]=255*(imghls[:][:,:][:,:,1]/normalization)
	imghls[:][:,:][:,:,2]=255*(imghls[:][:,:][:,:,2]/normalization)

	return imghls

#-- evalMultiGaussian ---------------------------------
#
# Deletes the pixels inside the specified contour
#
#-------------------------------------------------
def evalMultiGaussian(mean,cova,point):
	dimension = len(point)
	mean=array(mean)
	point=array(point)
	detCov = sqrt(linalg.det(cova))
	frac = (2*pi)**(-dimension/2.0) * (1/detCov)
	fprime = (mean - point)**2
	return frac * exp(-0.5*dot(dot(fprime, linalg.inv(cova)),fprime.T))

#-- Process ---------------------------------
#
# Saves process if flag turned on
#
#-------------------------------------------------
def Process(msg,img):
	if PROCESS:
		print msg
		if img != None:
			img2=img.copy()
			putText(img2,msg,(50,50),FONT_HERSHEY_COMPLEX_SMALL,2,(100,100,100))		
			imwrite("test/"+msg+image,img2)



#---------------------------------------Dan-R.           
#              |\/|  /\  | |\|               *
#              |  | /  \ | | \               *
#--------------------------------------------*

if __name__ == '__main__':

	image=sys.argv[1]
	
	labels=["Scalpel","Hook","Forceps","Clipper","Scissors","Hemostat","Retractor"]

	out=executeUnix("ls,images")
	#instrumentsets is the number of one image
	instrumentsets=out.split("\n")



	#start= time.clock()#-------------------------------------------STAGE 1: PREPROCESSING

	Process("Processing Image "+image,None)
	img = imread('test/'+image)
	img = medianBlur(img,5)

	try:
		impre = bilateralFilter(img,7,7*2,7/2)
	except:
		print "The image ",image,"is not in the /test directory"
		sys.exit()
	Process("1a.Preprocessed",impre)	

	imgnormal=normalize(impre.copy())
	Process("1b.Normalized",imgnormal)

	imghls=cvtColor(impre, cv.CV_RGB2HLS)
	imghls[:][:,:][:,:,1]=0;
	Process("1c.HLS",imghls)

	imgcanny=Canny(imghls,50,100)	
	#imgclean=deleteAllBlobs(imgcanny,BLOBNOISE)
	element = getStructuringElement(MORPH_RECT,(3,3))	
	imgcanny=morphologyEx(imgcanny,MORPH_DILATE,element,iterations=1)
	Process("1d.Canny",imgcanny)

	imgbin=morphologyEx(imgcanny,MORPH_CLOSE,element,iterations=MORPHOLOGYCLOSINGS)
	#imgbin=morphologyEx(imgcanny,MORPH_ERODE,element2,iterations=MORPHOLOGYCLOSINGS)
	Process("1e.FinalBinary",imgbin)


	#start= time.clock()#-------------------------------------------------------------
	#endtime()#-----------------------------------------------------STAGE 2: FIND BLOBS

	Process('2.FindBlobs',None)		
	blobs=AllBlobs(imgbin)

	#start= time.clock()#-------------------------------------------------------------
	#endtime()#-------------------------------------------------STAGE 3: SEGMENT IMAGES

	#** Find Maximum Blob
	maxlen=0
	for k,v in blobs.iteritems():
		if(len(blobs[k])>maxlen):
			maxlen=len(blobs[k])
			maxlenkey=k

	covm=getBlobCov(blobs,maxlenkey,imgnormal)
	backgroundmean=getBlobMean(blobs,maxlenkey,imgnormal)

	imgseg=255*ones(imgbin.shape)
	imback=255*ones(imgbin.shape)

	#**Generate Segmented image
	for k,v in blobs.iteritems():
		tempmean=getBlobMean(blobs,k,imgnormal)
		prob=evalMultiGaussian(backgroundmean,covm,tempmean)
		if DEBUG: 	print "foFr key",k,"the distance is:",prob

		if(prob>SEGMENTATIONPROB):
			for i,j in blobs[k]:
				imgseg[i][j]=0

	#**Generate Background image
	for i,j in blobs[maxlenkey]:
		imback[i][j]=0

	Process("3a.SegmentedImage",imgseg)
	Process("3b.SegmentedBackground",imback)

	imback=asarray(imback,dtype=uint8)

	#start= time.clock()#-------------------------------------------------------------
	#endtime()#--------------------------------------------------STAGE 4: FIND CONTOURS

	contours, hierarchy2 = findContours(imback.copy(),RETR_LIST,CHAIN_APPROX_SIMPLE)

	#start= time.clock()#--------------------------------------------------------------
	#endtime()#---------------------------------------------------------STAGE 5: TEST!
	
	i=0
	for cnt in contours:
		if(len(cnt)>MINBLOBSIZE):	

			x,y,w,h = boundingRect(cnt)
			
			a=cnt
			#Draw the bounding box of this instrument
			((x,y),(width,height),theta) = minAreaRect(a)
			box = cv.BoxPoints(((x,y),(width,height),theta))
			box = int0(box)	
			line(img, tuple(box[0]), tuple(box[1]),(17,17,111),2)
			line(img, tuple(box[1]), tuple(box[2]),(17,17,111),2)
			line(img, tuple(box[2]), tuple(box[3]),(17,17,111),2)
			line(img, tuple(box[3]), tuple(box[0]),(17,17,111),2)



			#**Draw contours in a larger image so that when you rotate you dont go out of bounds
			instrument=zeros((2*imback.shape[0],2*imback.shape[1]))
			drawContours(instrument,[cnt],0,255,-1)
			t=array([[1,0,instrument.shape[0]*0.1],[0,1,instrument.shape[1]*0.1]])
			instrument=warpAffine(instrument,t,(instrument.shape[1],instrument.shape[0]))

			#**Normalize scale and rotation	
			u,center=axes(instrument,cnt)
			imout=normalizeRotation(instrument,u,center[0]+instrument.shape[0]*0.1,center[1]+instrument.shape[1]*0.1)
			imout=asarray(imout,dtype=uint8)

			#Draw the major and minor axis
			line(img,(int(center[0]+u[0][0]*200),int(center[1]+u[0][1]*200)),(int(center[0]-u[0][0]*200),int(center[1]-u[0][1]*200)),(11,17,17),2)
			line(img,(int(center[0]+u[1][0]*50),int(center[1]+u[1][1]*50)),(int(center[0]-u[1][0]*50),int(center[1]-u[1][1]*50)),(111,17,17),2)

			#**Find final contour of instrument 
			fcontours, fhierarchy2 = findContours(imout.copy(),RETR_LIST,CHAIN_APPROX_SIMPLE)

			if len(fcontours)>1:
				print "Warning: two contours were found"
				maxi=0
				for i,cnt in enumerate(fcontours):
					if(len(cnt)>maxi):
						maxidx=i
				fcnt=fcontours[i]
			else:
				fcnt=fcontours[0]



			#**Normalize the final contour and image to position and store them as features
			imout,fcnt=NormContourToPosition(imout,fcnt)
			
	
			#**Automatically label the features\
			if INTERACTIVE:				
				imshow("Instrument", imout)
				if(waitKey(200)==27):
					break
				label=input("Press to close window ")
				destroyAllWindows()

			results=[]
			results2=[]
			for instrumentset in instrumentsets:
				instrumentset=instrumentset.strip(".JPG")
				result=[]
				result2=[]
				for trainins in range(28):
					
					b=load("storage/cnt"+instrumentset+"_"+str(trainins)+".npy")
					res=matchShapes(a,b,cv.CV_CONTOURS_MATCH_I3,0)
					result2.append(res)
					#Find NCC
					signature1=findSignature(a)
					signature2=findSignature(b)
					res=ncc2(signature1,signature2)
					result.append(res)

				if len(result)>0:
					results.append(result)
					results2.append(result2)
			results=array(results)
			results2=array(results2)
		
			votes2=results2.argmin(axis=1)%7
			#weights2=results2.min(axis=1)	
			votes=results.argmax(axis=1)%7

			d={}
			for v1,v2 in zip(votes2,votes):	
				d[v1]=d.get(v1,0)+1
				d[v2]=d.get(v2,0)+1
	
			maxval=0
			maxkey=0
			for k,v in zip(d.keys(),d.values()):
				if v>maxval:
					maxval=v
					maxkey=k

			
	
			#** Print the classificattion decision
			print labels[maxkey]
			#** Print the evidence of the classification
			print d
			#Finally label the instrument in the image
			putText(img,labels[maxkey],(center[0],center[1]),FONT_HERSHEY_PLAIN,4,(11,11,11))	
			

			i=i+1

	imwrite("test/output"+image,img)



	




