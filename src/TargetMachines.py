from lstar_extraction.Moore import Moore

unary_alphabet = ["0"]
bin_alphabet = ["0", "1"]

accepting_states = {
     "dyck_1" : [""],
     "dyck_2" : [""],
     "grid_2" : [""],
     "grid_3" : [""],
     "first" : ["1"],
     "parity" : [""],
     "ones" : [""]
}

"""
Depth-bounded Dyck languages
"""
# (10)* 
dyck_1 = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "1", "0"], 
                                  q0="",
                                  delta={"": {"1": "1", "0": "0"}, "0": {"0": "1", "1": ""}, "1": {"0": "1", "1": "1"}},
                                  X={"":0, "1":2, "0":1}
                                  )

dyck_1_dfa= Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "1", "0"], 
                                  q0="",
                                  delta={"": {"1": "1", "0": "0"}, "0": {"0": "1", "1": ""}, "1": {"0": "1", "1": "1"}},
                                  X={"":1, "1":0, "0":0}
                                  )

# character prediction binary encoding: first bit is omega valid (accepting state), second bit is 0 valid, third bit is 1 valid
dyck_1_char_pred = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "1", "0"], 
                                  q0="",
                                  delta={"": {"1": "1", "0": "0"}, "0": {"0": "1", "1": ""}, "1": {"0": "1", "1": "1"}},
                                  X={"":6, "1":0, "0":1}   # 110 000 001
                                  )

# (1(10)*0)* 
dyck_2 = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "1", "0", "00"], 
                                  q0="",
                                  delta={"": {"1": "1", "0": "0"}, "0": {"0": "00", "1": ""}, "1": {"0": "1", "1": "1"}, "00": {"0": "1", "1": "0"}},
                                  X={"":0, "0":1, "00": 2, "1": 3}
                                  )

dyck_2_dfa = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "1", "0", "00"], 
                                  q0="",
                                  delta={"": {"1": "1", "0": "0"}, "0": {"0": "00", "1": ""}, "1": {"0": "1", "1": "1"}, "00": {"0": "1", "1": "0"}},
                                  X={"":1, "0":0, "00": 0, "1": 0}
                                  )

dyck_2_char_pred = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "1", "0", "00"], 
                                  q0="",
                                  delta={"": {"1": "1", "0": "0"}, "0": {"0": "00", "1": ""}, "1": {"0": "1", "1": "1"}, "00": {"0": "1", "1": "0"}},
                                  X={"":6, "0":3, "00": 1, "1": 0}      # 110 011 001 000
                                  )

"""
Gridworld 
"""
grid_2 = Moore.initialize_static(alphabet=bin_alphabet,
                                  Q=["","1"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0":"", "1": "1"}},
                                  X={"":0, "1":1}
                                  )

grid_2_dfa = Moore.initialize_static(alphabet=bin_alphabet,
                                  Q=["","1"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0":"", "1": "1"}},
                                  X={"":1, "1":0}
                                  )

grid_2_char_pred = Moore.initialize_static(alphabet=bin_alphabet,
                                  Q=["","1"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0":"", "1": "1"}},
                                  X={"":7, "1":3} # 111 011
                                  )

grid_3 = Moore.initialize_static(alphabet=bin_alphabet,
                                  Q=["","1","11"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0":"", "1": "11"}, "11":{"0":"1", "1":"11"}},
                                  X={"":0, "1":1, "11":2}
                                  )

grid_3_dfa = Moore.initialize_static(alphabet=bin_alphabet,
                                  Q=["","1","11"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0":"", "1": "11"}, "11":{"0":"1", "1":"11"}},
                                  X={"":1, "1":0, "11":0}
                                  )

grid_3_char_pred = Moore.initialize_static(alphabet=bin_alphabet,
                                  Q=["","1","11"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0":"", "1": "11"}, "11":{"0":"1", "1":"11"}},
                                  X={"":7, "1":3, "11":3} # 111 011 011
                                  )

"""
Other
"""
first = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["","0","1"], 
                                  q0="",
                                  delta={"":{"0":"0", "1": "1"}, "1":{"0": "1", "1": "1"}, "0":{"0": "0", "1": "0"}},
                                  X={"":0, "0":1, "1":2}
                                  )

first_dfa = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["","0","1"], 
                                  q0="",
                                  delta={"":{"0":"0", "1": "1"}, "1":{"0": "1", "1": "1"}, "0":{"0": "0", "1": "0"}},
                                  X={"":0, "0":0, "1":1}
                                  )

first_char_pred = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["","0","1"], 
                                  q0="",
                                  delta={"":{"0":"0", "1": "1"}, "1":{"0": "1", "1": "1"}, "0":{"0": "0", "1": "0"}},
                                  X={"":1, "0":0, "1":7} # 001 000 111
                                  )

# 1*
ones = Moore.initialize_static(alphabet=bin_alphabet,
                                  Q=["", "0"],
                                  q0="",
                                  delta={"":{"0":"0", "1": ""}, "0":{"0":"0", "1": "0"}},
                                  X={"":0, "0":1}
                                  )

ones_dfa = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "0"], 
                                  q0="",
                                  delta={"":{"0":"0", "1": ""}, "0":{"0":"0", "1": "0"}},
                                  X={"":1, "0":0}
                                  )

ones_char_pred = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "0"], 
                                  q0="",
                                  delta={"":{"0":"0", "1": ""}, "0":{"0":"0", "1": "0"}},
                                  X={"":5, "0":0}     # 101 000
                                  )

parity = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["","1"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0": "1", "1": ""}},
                                  X={"":0, "1":1}
                                  )

parity_dfa = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["","1"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0": "1", "1": ""}},
                                  X={"":1, "1":0}
                                  )

parity_char_pred = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["","1"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0": "1", "1": ""}},
                                  X={"":7, "1":3} # 111 011
                                  )

def print_all():
    print("Printing all")

    dyck_1.draw_nicely(name="dyck_1")
    dyck_1_dfa.draw_nicely(name="dyck_1_dfa")
    dyck_1_char_pred.draw_nicely(name="dyck_1_char_pred")

    dyck_2.draw_nicely(name="dyck_2")
    dyck_2_dfa.draw_nicely(name="dyck_2_dfa")
    dyck_2_char_pred.draw_nicely(name="dyck_2_char_pred")

    grid_2.draw_nicely(name="grid_2")
    grid_2_dfa.draw_nicely(name="grid_2_dfa")
    grid_2_char_pred.draw_nicely(name="grid_2_char_pred")

    grid_3.draw_nicely(name="grid_3")
    grid_3_dfa.draw_nicely(name="grid_3_dfa")
    grid_3_char_pred.draw_nicely(name="grid_3_char_pred")

    first.draw_nicely(name="first")
    first_dfa.draw_nicely(name="first_dfa")
    first_char_pred.draw_nicely(name="first_char_pred")

    parity.draw_nicely(name="parity")
    parity_dfa.draw_nicely(name="parity_dfa")
    parity_char_pred.draw_nicely(name="parity_char_pred")

def get_target_moore(language_name, training_task="state_prediction"):
      if training_task == "membership_prediction":
           return globals()[language_name+"_dfa"]
      elif training_task == "character_prediction":
            return globals()[language_name+"_char_pred"]
      else:
            return globals()[language_name]

if __name__ == "__main__":
    print_all()