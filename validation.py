from cv2 import *
from numpy import *
from mysys import *
from signature import *

import operator

PROCEDURE=False

if __name__ == '__main__':

	confusion=zeros((7,7))	
	
	for testset in range(5,24):

		out=executeUnix("ls,images")
		#out=executeUnix("ls,storages")
		instrumentsets=out.split("\n")

		#** Test all test instruments for all instruments in the instrument set
		for testins in range(7):
			a=load("storage/cnt"+str(testset)+"_"+str(testins)+".npy")
			results=[]
			results2=[]
			for instrumentset in instrumentsets:
				instrumentset=instrumentset.strip(".JPG")
				result=[]
				result2=[]
				for trainins in range(28):
					#**Avoid testing with trained data
					if(str(testset)!=instrumentset):
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
		
			#** Note that testins corresponds to the expected result
			if PROCEDURE:	print maxkey,testins
			if PROCEDURE:	print d
		
			
			confusion[maxkey][testins]+=1
			if PROCEDURE:	print confusion

	print confusion
		


		
			
		
	









