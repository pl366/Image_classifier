from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.files.storage import FileSystemStorage
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import os
import keras.backend
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import shutil  
p=100
# Create your views here.
def classifier(request):
	keras.backend.clear_session()
	if (request.method == 'POST' and request.FILES['myfile']):
		myfile = request.FILES['myfile']
		fs = FileSystemStorage()
		filename = fs.save(myfile.name, myfile)
		uploaded_file_url = fs.url(filename)
		print (type(uploaded_file_url))
		uploaded_file_url=uploaded_file_url.replace('%20',' ')
		print (uploaded_file_url)
		model_path = '../Desktop/models/classifier.h5'
		model_weights_path = '../Desktop/models/weights.h5'
		model = load_model(model_path)
		model.load_weights(model_weights_path)
		test_image = load_img(os.getcwd()+uploaded_file_url, target_size = (64, 64))
		test_image = img_to_array(test_image)
		print (test_image.shape)
		test_image = np.expand_dims(test_image, axis = 0)
		result = model.predict(test_image)
		print (result)
		answer = np.argmax(result[0])
		Confidence= result[0][answer]
		
		result=""
		if answer == 0:
			result="aadharcard"
			print("Label: aadharcard")
		elif answer == 1:
			result="pancard"
			print("Labels: pancard")
		elif answer == 2:
			result="passport"
			print("Label: passport")
		elif answer ==3:
			result="votercard"
			print("Label: votercard")
		elif answer ==4:
			result="vzjunk"
			print("Label: vzjunk") 	 
		return render(request, 'classifier/index.html', {'result': result,'conf':Confidence,'uploaded_url':uploaded_file_url})



	return render(
		request, 'classifier/index.html', {}
		)

def training(request):
	keras.backend.clear_session()
  	
	url =''
	if request.GET:
		url = request.GET['url']
	print (type(url))	

	if (request.method == 'POST'):
		a= request.POST['id']
		print ("dytu",a)
	l=url.rsplit('/', 1)[-1]
	s=(l.rsplit('.',1)[-1])
	# print (os.getcwd()+url)
	# print ('../Desktop/Dataset/training_set/'+a+'/'+p)
	# url=url.replace(" ", "")
	# print ('url')
	global p
	shutil.copyfile(os.getcwd()+url,'../Desktop/Dataset/training_set/'+a+'/image'+str(p)+'.'+s)
	# print ('../Desktop/Dataset/training_set/'+id+'/'+p)		
	p+=1
	classifier =Sequential()
	classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2, 2)))
	classifier.add(Flatten())
	classifier.add(Dense(units = 128, activation = 'relu'))
	classifier.add(Dense(units = 5, activation = 'softmax'))
	classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

	train_datagen = ImageDataGenerator(rescale = 1./255,
	rotation_range = 60,
	shear_range = 0.2,
	zoom_range = 0.2,
	horizontal_flip = True)

	test_datagen = ImageDataGenerator(rescale = 1./255)

	training_set = train_datagen.flow_from_directory('../Desktop/Dataset/training_set',
	target_size = (64, 64),
	batch_size = 32,
	class_mode = 'categorical')

	test_set = test_datagen.flow_from_directory('../Desktop/Dataset/test_set',
	target_size = (64, 64),
	batch_size = 32,
	class_mode = 'categorical')

	classifier.fit_generator(training_set,
	steps_per_epoch = 70,
	epochs = 3,
	validation_data = test_set,
	validation_steps = 16)


	target_dir = '../Desktop/models/'
	if not os.path.exists(target_dir):
	  os.mkdir(target_dir)
	classifier.save('../Desktop/models/classifier.h5')
	classifier.save_weights('../Desktop/models/weights.h5')


	return HttpResponseRedirect('/')