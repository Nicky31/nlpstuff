import os
import mmap
import multiprocessing as mp
from threading import Thread
from itertools import islice
from tqdm import tqdm
from .Preprocessing import Preprocessing

"""
	If :use_cache and a cached training file is available for dataset, returns it
	Otherwise, preprocess dataset & write result in a cached training file
	Returns training file filepath
"""
def get_training_file(
	dataset, use_cache=True, batch_size=1,
	lemmatize=True, remove_stopwords=True,
	progressbar=False, tokenizer="spacy",
	num_workers=4
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

	# Prepare output queue threaded writting
	m = mp.Manager()
	output_queue = m.Queue()
	training_file_writting = Thread(target=threaded_writting, args=[out_filepath, output_queue])
	training_file_writting.start()

	# Run parallelized dataset file preprocessing
	parallelized_preprocessing(
		dataset_file=dataset.filepath,
		lang=dataset.lang,
		lemmatize=lemmatize,
		remove_stopwords=remove_stopwords,
		tokenizer=tokenizer,
		num_workers=num_workers,
		output_queue=output_queue,
		progressbar=progressbar
	)

	# Wait for output buffer writting
	output_queue.put("__END_OF_CORPUS__")
	training_file_writting.join()

	print("Preprocessed file generated at '{}'".format(out_filepath))
	return out_filepath

"""
	Spawns a processes pool to preprocess lines from dataset_file, store result in output_queue
	Do not return before entire dataset_file preprocessing
"""
def parallelized_preprocessing(
	dataset_file, lang, lemmatize,
	remove_stopwords, tokenizer, num_workers,
	output_queue, progressbar
):
	preprocessing = Preprocessing(
		lang,
		lemmatize=lemmatize,
		remove_stopwords=remove_stopwords,
		tokenizer=tokenizer
	)

	pool = mp.Pool(num_workers if num_workers is not None else 4)
	jobs = []
	# Divide training file in chunk and distribute them to the processes pool
	chunks = list(chunkify(dataset_file))
	for chunk_start, chunk_size in chunks:
		jobs.append(pool.apply_async(preprocess_wrapper, (dataset_file, preprocessing, chunk_start, chunk_size, output_queue)))

	if progressbar:
		pbar = tqdm(total=len(jobs)) # prints progression in chunks
	# Wait for all lines to be preprocessed
	for job in jobs:
		job.get()
		if progressbar:
			pbar.update(1)

	pool.close()
	if progressbar:
		pbar.close()

def chunkify(filepath, size=1024*256):
	filesize = os.path.getsize(filepath)
	with open(filepath, 'rb') as f:
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
	with open(dataset_path, "rb") as f:
		f.seek(chunk_start)
		lines = f.read(chunk_size).decode().splitlines()
		output_queue.put(preprocessing(lines))
	
def threaded_writting(training_file, queue):
	with open(training_file, "w") as output_f:
		while True:
			lines = queue.get()
			if lines == "__END_OF_CORPUS__":
				break
			output_f.writelines(lines)