from lstar_extraction.Moore import Moore

unary_alphabet = ["0"]
bin_alphabet = ["0", "1"]

accepting_states = {
     "tomita_1" : [""],
     "tomita_2" : [""],
     "tomita_3" : ["", "1", "100"],
     "tomita_4" : ["", "0", "00"],
     "tomita_5" : [""],
     "tomita_6" : [""],
     "tomita_7" : ["", "1", "10", "101"],
     "dyck_1" : [""],
     "dyck_2" : [""],
     "grid_1" : [""],
     "grid_2" : [""],
     "grid_3" : [""],
     "length_2" : [""],
     "length_3" : [""],
     "length_4" : [""],
     "first" : ["1"],
     "parity" : [""],
     "ones" : [""]
}

"""
Tomita Languages
"""
# 1* = Ones
tomita_1 = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "0"], 
                                  q0="",
                                  delta={"":{"0":"0", "1": ""}, "0":{"0":"0", "1": "0"}},
                                  X={"":0, "0":1}
                                  )

# (10)* 
tomita_2 = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "1", "0"], 
                                  q0="",
                                  delta={"": {"1": "1", "0": "0"}, "0": {"0": "0", "1": "0"}, "1": {"0": "", "1": "0"}},
                                  X={"":0, "1":1, "0":2}
                                  )

# all strings without containing odd number of consecutive 0's after odd number of consecutive 1's
tomita_3 = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "1", "10", "101", "100"], 
                                  q0="",
                                  delta={"": {"0": "", "1": "1"}, 
                                        "1": {"0": "10", "1": ""}, 
                                        "10": {"0": "100", "1": "101"}, 
                                        "101": {"0": "101", "1": "101"}, 
                                        "100": {"0": "10", "1": "100"}},
                                  X={"":0, "1":1, "10":2, "101":3, "100":4}
                                  )

# all strings without containing 3 consecutive 0's (000)
tomita_4 = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["","0","00","000"], 
                                  q0="",
                                  delta={"": {"0": "0", "1": ""},
                                        "0": {"0": "00", "1": ""},
                                        "00": {"0": "000", "1": ""},
                                        "000": {"0": "000", "1": "000"}},
                                  X={"":0, "0":1, "00":2, "000":3}
                                  )

# all strings with even numbers of 0's and 1's
tomita_5 = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "0", "1", "01"], 
                                  q0="",
                                  delta={"": {"0": "0", "1": "1"},
                                        "0": {"0": "", "1": "01"},
                                        "1": {"0": "01", "1": ""},
                                        "01": {"0": "1", "1": "0"}},
                                  X={"":0, "0":1, "1":2, "01":3}
                                  )

# all strings satisfying #(0)-#(1) = 3n (n=...,-1,0,1,...)
tomita_6 = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "0", "1"], 
                                  q0="",
                                  delta={"": {"0": "0", "1": "1"},"0": {"0": "1", "1": ""},"1": {"0": "", "1": "0"}},
                                  X={"":0, "1":1, "0":2}
                                  )

# 1*0*1*0*
tomita_7 = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "1", "10", "101", "1010"], 
                                  q0="",
                                  delta={"": {"0": "", "1": "1"}, 
                                        "1": {"0": "10", "1": "1"}, 
                                        "10": {"0": "10", "1": "101"}, 
                                        "101": {"0": "1010", "1": "101"}, 
                                        "1010": {"0": "1010", "1": "1010"}},
                                  X={"":0, "1":1, "10":2, "101":3, "1010": 4}
                                  )

"""
Depth-bounded Dyck languages
"""
# (10)* 
dyck_1 = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "1", "0"], 
                                  q0="",
                                  delta={"": {"1": "1", "0": "0"}, "0": {"0": "1", "1": ""}, "1": {"0": "1", "1": "1"}},
                                  X={"":0, "1":1, "0":2}
                                  )

dyck_1_dfa= Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "1", "0"], 
                                  q0="",
                                  delta={"": {"1": "1", "0": "0"}, "0": {"0": "1", "1": ""}, "1": {"0": "1", "1": "1"}},
                                  X={"":1, "1":0, "0":0}
                                  )

dyck_1_char_pred = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["", "1", "0"], 
                                  q0="",
                                  delta={"": {"1": "1", "0": "0"}, "0": {"0": "1", "1": ""}, "1": {"0": "1", "1": "1"}},
                                  X={"":2, "1":1, "0":0}   # 10 01 00 
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
                                  X={"":2, "0":3, "00": 1, "1": 0}      # 10 11 01 00
                                  )

"""
Gridworld 
"""
grid_1 = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["","1"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0":"", "1": "1"}},
                                  X={"":0, "1":1}
                                  )

grid_1_dfa = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["","1"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0":"", "1": "1"}},
                                  X={"":1, "1":0}
                                  )

grid_1_char_pred = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["","1"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0":"", "1": "1"}},
                                  X={"":3, "1":3} # 11 11
                                  )

grid_2 = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["","1","11"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0":"", "1": "11"}, "11":{"0":"1", "1":"11"}},
                                  X={"":0, "1":1, "11":2}
                                  )

grid_2_dfa = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["","1","11"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0":"", "1": "11"}, "11":{"0":"1", "1":"11"}},
                                  X={"":1, "1":0, "11":0}
                                  )

grid_2_char_pred = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["","1","11"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0":"", "1": "11"}, "11":{"0":"1", "1":"11"}},
                                  X={"":3, "1":3, "11":3} # 11 11 11
                                  )

grid_3 = Moore.initialize_static(alphabet=bin_alphabet, 
                                  Q=["","1","11", "111"], 
                                  q0="",
                                  delta={"":{"0":"", "1": "1"}, "1":{"0":"", "1": "11"}, "11":{"0":"1", "1":"111"}, "111":{"0":"11","1":"111"}},
                                  X={"":0, "1":1, "11":2, "111":3}
                                  )

"""
LENGTH(q) 
"""
# (00)*
length_2 = Moore.initialize_static(alphabet=unary_alphabet, 
                                  Q=["","0"], 
                                  q0="",
                                  delta={"": {"0": "0"},"0": {"0": ""}},
                                  X={"":0, "0":1}
                                  )

# (000)*
length_3 = Moore.initialize_static(alphabet=unary_alphabet, 
                                  Q=["","0","00"], 
                                  q0="",
                                  delta={"": {"0": "0"},"0": {"0": "00"},"00":{"0":""}},
                                  X={"":0, "0":1, "00":2}
                                  )

# (0000)*
length_4 = Moore.initialize_static(alphabet=unary_alphabet, 
                                  Q=["","0","00","000"], 
                                  q0="",
                                  delta={"": {"0": "0"},"0": {"0": "00"},"00":{"0":"000"}, "000":{"0":""}},
                                  X={"":0, "0":1, "00":2, "000":3}
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
                                  X={"":0, "0":0, "1":1} # 01 00 11
                                  )

# 1*
ones = tomita_1

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
                                  X={"":1, "0":0}     # 01 00
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
                                  X={"":3, "1":3} # 11 11
                                  )

def print_all():
    print("Printing all")
    tomita_1.draw_nicely()
    tomita_2.draw_nicely()
    tomita_3.draw_nicely()
    tomita_4.draw_nicely()
    tomita_5.draw_nicely()
    tomita_6.draw_nicely()
    tomita_7.draw_nicely()

    dyck_1.draw_nicely(name="dyck_1")
    dyck_1_dfa.draw_nicely(name="dyck_1_dfa")
    dyck_2.draw_nicely()

    grid_1.draw_nicely(name="grid_1")
    grid_1_dfa.draw_nicely(name="grid_1_dfa")
    grid_2.draw_nicely()
    grid_3.draw_nicely()
    
    length_2.draw_nicely()
    length_3.draw_nicely()
    length_4.draw_nicely()
    
    first.draw_nicely()
    parity.draw_nicely(name="parity")
    parity_dfa.draw_nicely(name="parity_dfa")

def get_target_moore(language_name, training_task="state_prediction"):
      if training_task == "membership_prediction":
           return globals()[language_name+"_dfa"]
      elif training_task == "character_prediction":
            return globals()[language_name+"_char_pred"]
      else:
            return globals()[language_name]

