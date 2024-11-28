from time import perf_counter
from lstar_extraction.ObservationTable import TableTimedOut
from lstar_extraction.Moore import Moore
from lstar_extraction.Teacher import Teacher
from lstar_extraction.Lstar import run_lstar

def extract(oracle,time_limit = 50,initial_split_depth = 10,starting_examples=None):
	print("provided counterexamples are:",starting_examples)
	guided_teacher = Teacher(oracle,num_dims_initial_split=initial_split_depth,starting_examples=starting_examples)
	start = perf_counter()
	try:
		run_lstar(guided_teacher,time_limit)
	except KeyboardInterrupt: #you can press the stop button in the notebook to stop the extraction any time
		print("lstar extraction terminated by user")
	except TableTimedOut:
		print("observation table timed out during refinement")
	end = perf_counter()
	extraction_time = end-start

	dfa = guided_teacher.dfas[-1]

	print("overall guided extraction time took: " + str(extraction_time))

	print("generated counterexamples were: (format: (counterexample, counterexample generation time))")
	print('\n'.join([str(a) for a in guided_teacher.counterexamples_with_times]))

	return dfa, extraction_time