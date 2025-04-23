from lstar_extraction.ObservationTable import ObservationTable, TableTimedOut
from extraction.Moore import Moore
from time import perf_counter
from extraction.Teacher import Teacher

def run_lstar(teacher,time_limit):
	table = ObservationTable(teacher.alphabet,teacher)
	start = perf_counter()
	teacher.counterexample_generator.set_time_limit(time_limit,start)
	table.set_time_limit(time_limit,start)

	while True:
		while True:
			while table.find_and_handle_inconsistency():
				pass
			if table.find_and_close_row():
				continue
			else:
				break
		dfa = Moore(obs_table=table)
		print("obs table refinement took " + str(int(1000*(perf_counter()-start))/1000.0) )
		counterexample = teacher.equivalence_query(dfa)
		if None is counterexample:
			break
		start = perf_counter()
		table.add_counterexample(counterexample,teacher.classify_word(counterexample))
	return dfa

def extract(transformer, time_limit=50, initial_split_depth = 10, starting_examples=None):
	print("provided counterexamples are:",starting_examples)
	guided_teacher = Teacher(transformer,num_dims_initial_split=initial_split_depth,starting_examples=starting_examples)
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