from cv2 import *
from numpy import *
import operator
import itertools
import time
from scipy.spatial.distance import mahalanobis
import sys
from mysys import *
from myblobs import *


MINSIZECONTOUR=100
SEGMENTATIONPROB=0

MORPHOLOGYCLOSINGS=3
Toy=False
PROCESS=True
DEBUG=False
BLOBNOISE=20
MINBLOBSIZE=100
#MINSIZECONTOUR=

#-- Normalize ---------------------------------
#
# Deletes the pixels inside the specified contour
#
#-------------------------------------------------
def normalize(imghls):
	"""a=array([[1,5,5],[5,50,1],[1,1,5]])
	print a
	print a*(a!=5)+(a==5)*10
	#print (a==5)*10+a
	#imwrite("a.jpg",a)"""
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
			imwrite(image+"/"+msg+".jpg",img2)



#---------------------------------------Dan-R.           
#              |\/|  /\  | |\|               *
#              |  | /  \ | | \               *
#--------------------------------------------*

if __name__ == '__main__':
	out=executeUnix("ls,images")
	images=out.split("\n")

	for image in images:
		image=image.strip(".JPG")
	
		#start= time.clock()#-------------------------------------------------------------
		if Toy:
			imgin=imread("toy.jpg")
			imgbin=cvtColor(imgin, cv.CV_RGB2GRAY);

			#Make sure the image is binary
			for i in range(imgbin.shape[0]):
				for j in range(imgbin.shape[1]):
					if(imgbin[i][j]<125):
						imgbin[i][j]=0
					else:
						imgbin[i][j]=255

		if not Toy:
		
			image='20'
			Process("Processing Image "+image+".JPG",None)
			img = imread('images/'+image+'.JPG')
			img = medianBlur(img,5)

			impre = bilateralFilter(img,7,7*2,7/2)
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
		#endtime()#-----------------------------------------------------------------------

		Process('2.FindBlobs',None)		
		blobs=AllBlobs(imgbin)

		#start= time.clock()#-------------------------------------------------------------
		#endtime()#-----------------------------------------------------------------------

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
			if DEBUG: 	print "for key",k,"the distance is:",prob

			if(prob>SEGMENTATIONPROB):
				for i,j in blobs[k]:
					imgseg[i][j]=0

		#**Generate Background image
		for i,j in blobs[maxlenkey]:
			imback[i][j]=0
		
		Process("3a.SegmentedImage",imgseg)
		Process("3b.SegmentedBackground",imback)

		#dest = cv.LoadImageM(image+'/3b.SegmentedBackground.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)

		imback=asarray(imback,dtype=uint8)
	
		#start= time.clock()#-------------------------------------------------------------
		#endtime()#-----------------------------------------------------------------------

		contours, hierarchy = findContours(imback.copy(),RETR_LIST,CHAIN_APPROX_SIMPLE)

		#finalcontours=[]
		o=0
		instruments=[]
		for cnt in contours:
			if(len(cnt)<MINBLOBSIZE):	
				deleteBlobFromContour(imback,cnt)
			else:
				M = moments(cnt)
				centroid_x = int(M['m10']/M['m00'])
				centroid_y = int(M['m01']/M['m00'])
				print cnt
	 			
				cnt=reshape(cnt,(cnt.shape[0],2))
		
				print cov(cnt.T)
			

				#print cov(cnt.T)
				#print "done"
				#print cov(cnt)
				#x,y,w,h = boundingRect(cnt)
				#e = fitEllipse(cnt)
				#ellipse(impre,e,(151,50,106),2)
				#Process("4.BoundingBox"+str(o),impre)
				"""instrument=zeros(imback.shape)
				drawContours(instrument,[cnt],0,255,-1)
				instrument=instrument[y:y+h][:,x:x+w]
				instruments.append(instrument)
				Process("Instrument"+str(o),instrument)
				instrument=asarray(instrument,dtype=uint8)
				contours, hierarchy = findContours(instrument.copy(),RETR_LIST,CHAIN_APPROX_SIMPLE)
				for c in contours:
					e = fitEllipse(c)
					ellipse(impre[y:y+h][:,x:x+w],e,(151,50,106),2)
					Process("4.BoundingBox"+str(o),impre[y:y+h][:,x:x+w])"""
			
				#Process("Instrument"+str(o),imback[y:y+h][:,x:x+w])			
				#rectangle(impre,(x,y),(x+w,y+h),(151,50,106),1)
				"""((x,y),(width,height),theta) = minAreaRect(cnt)
				box = cv.BoxPoints(((x,y),(width,height),theta))
				box = int0(box)	
				line(impre, tuple(box[0]), tuple(box[1]),(17,17,111),2)
				line(impre, tuple(box[1]), tuple(box[2]),(17,17,111),2)
				line(impre, tuple(box[2]), tuple(box[3]),(17,17,111),2)
				line(impre, tuple(box[3]), tuple(box[0]),(17,17,111),2)"""
	
				o+=1
			

		Process("4.BoundingBox",impre)
	

		#endtime()#-----------------------------------------------------------------------


		sys.exit()





