import os
import mmap
import multiprocessing as mp
from threading import Thread
from itertools import islice
from tqdm import tqdm
from .Preprocessing import Preprocessing

def chunkify(filepath, size=1024*1024):
	filesize = os.path.getsize(filepath)
	with open(filepath, 'r') as f:
		chunk_end = f.tell()
		while True:
			chunk_start = chunk_end
			f.seek((chunk_start + size) if (chunk_start + size) < filesize else (filesize - 1))
			f.readline()
			chunk_end = f.tell()
			yield chunk_start, chunk_end - chunk_start
			if chunk_end >= (filesize - 1):
				break

def preprocess_wrapper(dataset_path, preprocessing, chunk_start, chunk_size, output_queue):
	print("preprocess chunk {} -> {}".format(chunk_start, chunk_start + chunk_size))
	with open(dataset_path) as f:
		f.seek(chunk_start)
		lines = f.read(chunk_size).splitlines()
		for line in lines:
			preprocessed = " ".join(preprocessing(line)) + "\n"			
			output_queue.put(preprocessed)
			
def threaded_writting(training_file, queue):
	with open(training_file, "w") as output_f:
		while True:
			line = queue.get()
			if line == "__END_OF_CORPUS__":
				break
			output_f.write(line)


"""
	If :use_cache and a cached training file is available for dataset, returns it
	Otherwise, preprocess dataset & write result in a cached training file
	Returns training file filepath
"""
def get_training_file(
	dataset, use_cache=True, batch_size=1,
	lemmatize=True, remove_stopwords=True,
	progressbar=False, tokenizer="spacy"
):
	out_filepath = dataset.cached_training_filepath
	if use_cache and os.path.exists(out_filepath):
		return dataset.cached_training_filepath
	
	# No cache,reate training file
	try:
		total_size = os.stat(dataset.filepath).st_size
		if total_size == 0:
			raise Exception()
	except:
		raise Exception("Missing or empty training file")
	if progressbar:
		pbar = tqdm(total=total_size) # prints progression in bytes
	# Link preprocessing generator to our cache file
	preprocessing = Preprocessing(
		dataset.lang,
		lemmatize=lemmatize,
		remove_stopwords=remove_stopwords,
		tokenizer=tokenizer
	)


	# with open(dataset.filepath, 'r') as input_f:
		# map_file = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
		# for line in iter(map_file.readline, b""):
		# for line in input_f:			
	
	pool = mp.Pool(3)
	jobs = []
	m = mp.Manager()
	output_queue = m.Queue()
	training_file_writting = Thread(target=threaded_writting, args=[out_filepath, output_queue])
	training_file_writting.start()
	for chunk_start, chunk_size in chunkify(dataset.filepath):
		print("chunk {} -> {}".format(chunk_start, chunk_start + chunk_size))
		jobs.append(pool.apply_async(preprocess_wrapper, (dataset.filepath, preprocessing, chunk_start, chunk_size, output_queue)))
		# output_f.write(" ".join(preprocessing(line)) + "\n")
		if progressbar:
			pbar.update(linelen)

	# Wait for all lines to be preprocessed
	for job in jobs:
		job.get()
	# Wait for output buffer writting
	output_queue.put("__END_OF_CORPUS__")
	training_file_writting.join()

	if progressbar:
		pbar.close()
	pool.close()
	print("Preprocessed file generated at '{}'".format(out_filepath))
	return out_filepath
