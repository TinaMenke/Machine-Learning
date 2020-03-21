# Import modules

import numpy as np
from scipy import misc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def read_scene():
	data_x = misc.imread('../../Data/Scene/times_square.jpg')

	return (data_x)


if __name__ == '__main__':
	
	################################################
	# K-Means

	data_x = read_scene()
	print('X = ', data_x.shape)

	flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
	print('Flattened image = ', flattened_image.shape)

	print('Implement k-means here ...')

	################################################

	k_values = [2, 5, 10, 25, 50, 75, 100, 200]
	N = 160000
	sum_squared_diff = []

	#plot configurations
	fig, axs = plt.subplots(ncols=3, nrows=3, figsize = (9,9))
	plt.subplots_adjust(wspace=0.3,hspace=0.3)
	reconstructed_image = flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
	axs[0, 0].imshow(reconstructed_image)
	axs[0,0].set_title("original image")
	l=1

	for k in k_values:
		differences = 0
		new_pic=[]
		new_pic_float=[]
		kmeans = KMeans(n_clusters=k, random_state=0).fit(flattened_image)
		centers = kmeans.cluster_centers_
		predictions = kmeans.predict(flattened_image)
		for i in np.arange(N):
			value = centers[predictions[i]]
			value_float = value
			value = [int(v) for v in value]
			new_pic.append(value)
			new_pic_float.append(value_float)
			# sum of squared distances of each pixel to its respective cluster centroids
			difference = (flattened_image[i][0]-new_pic_float[i][0])**2+(flattened_image[i][1]-new_pic_float[i][1])**2+(flattened_image[i][2]-new_pic_float[i][2])**2
			differences = differences + difference
		sum_squared_diff = np.append(sum_squared_diff, differences)
		new_pic = np.array(new_pic)
		print(differences)
		new_pic_reshape = new_pic.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
		
		#plot
		if(k==2):
			axs[0,1].imshow(new_pic_reshape)
			axs[0,1].set_title("k = 2")
		elif(k==5):
			axs[0,2].imshow(new_pic_reshape)
			axs[0,2].set_title("k = 5")
		elif(k==10):
			axs[1,0].imshow(new_pic_reshape)
			axs[1,0].set_title("k = 10")
		elif(k==25):
			axs[1,1].imshow(new_pic_reshape)
			axs[1,1].set_title("k = 25")
		elif(k==50):
			axs[1,2].imshow(new_pic_reshape)
			axs[1,2].set_title("k = 50")
		elif(k==75):
			axs[2,0].imshow(new_pic_reshape)
			axs[2,0].set_title("k = 75")
		elif(k==100):
			axs[2,1].imshow(new_pic_reshape)
			axs[2,1].set_title("k = 100")
		elif(k==200):
			axs[2,2].imshow(new_pic_reshape)
			axs[2,2].set_title("k = 200")

	plt.show()
	fig.savefig('../Figures/image_compression.png')

	
	#plt.semilogy(k_values,sum_squared_diff)
	plt.plot(k_values, sum_squared_diff)
	plt.title("Sum of squared errors")
	plt.xlabel("value of k")
	plt.ylabel("sum of squared error")
	plt.savefig('../Figures/error_plot.png')
	plt.show()







	################################################


	#reconstructed_image = flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
	print('Reconstructed image = ', reconstructed_image.shape)







