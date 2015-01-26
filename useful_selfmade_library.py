import random

def DataPartition(data,jupiter):
	# jupiter is a float that parses the length of data
	# data is what we want to split
	first = round(jupiter * len(data))

	random.shuffle(data)

	train_data = data[:first,:]
	test_data = data[first:,:]
	return(train_data,test_data)