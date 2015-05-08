from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
import csv
import time


with open('results.csv', 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	writer.writerow(['k', 'alg', 'train_size', 'test_size', 'Fit Time', 'Unit Fit Time', 'Test Time', 'Unit Test Time', 'Accuracy'])


	f = open('results.txt', 'w')

	# Print and write to storage file.
	def pw(text):
		print(text)
		f.write(text + "\n")


	# Loading in data
	pw("Loading galaxy data.")
	galaxy = io.loadmat("galaxy_grey.mat")

	train_data = galaxy['train_data']
	train_labels = galaxy['train_labels']
	val_data = galaxy['val_data']
	val_labels = galaxy['val_labels']
	test_data = galaxy['test_data']
	test_labels = galaxy['test_labels']
	pw("Finished loading galaxy data.\n")

	pw('k, alg, train_size, test_size, Fit Time, Unit Fit Time, Test Time, Unit Test Time, Accuracy\n')

	train_labels_argmax = np.argmax(train_labels, axis=1)
	val_labels_argmax = np.argmax(val_labels, axis=1)


	def run_knn(k, alg, train_size, test_size, dummy=False):

		if dummy:
			pw("Starting cpu conditioning dummy run.")

		knnclas = KNeighborsClassifier(n_neighbors=k, algorithm=alg)

		fit_start_time = time.time()
		knnclas.fit(train_data[:train_size], train_labels_argmax[:train_size])
		fit_end_time = time.time()
		fit_time = fit_end_time - fit_start_time
		fit_time_str = "{:.5f}".format(fit_time)
		#fit_time_str = str(round(fit_time, 3))
		unit_fit_time = fit_time / train_size
		unit_fit_str = "{:.8f}".format(unit_fit_time)
		#unit_fit_str = str(round(unit_fit_time, 5))


		test_start_time = time.time()
		accuracy = knnclas.score(val_data[:test_size], val_labels_argmax[:test_size])
		test_end_time = time.time()
		test_time = test_end_time - test_start_time
		test_time_str = "{:.5f}".format(test_time)
		#test_time_str = str(round(test_time, 3))
		unit_test_time = test_time / test_size
		unit_test_str = "{:.8f}".format(unit_test_time)
		#unit_test_str = str(round(unit_test_time, 5))

		accuracy_str = "{:.4f}".format(accuracy)
		#accuracy_str = str(round(accuracy, 3))

		if dummy:
			pw("Finished dummy run.\n")
		else:
			writer.writerow([str(k), alg, str(train_size), str(test_size), fit_time_str, unit_fit_str, test_time_str, unit_test_str, accuracy_str])
			pw(str(k) + ", " + alg + ", " + str(train_size) + ", " + str(test_size) + ", " + fit_time_str + ", " + unit_fit_str + ", " + test_time_str + ", " + unit_test_str + ", " + accuracy_str)


	k = 3
	alg = 'ball_tree'
	train_size = 10000
	test_size = 250

	params = [200]

	for i in xrange(49):
		params.append(params[-1] + 200)

	#Dummy run to condition cpu.
	run_knn(3, 'ball_tree', 5000, 500, dummy=True)	

	for param in params:
		run_knn(k, alg, param, test_size)
		csvfile.flush()


