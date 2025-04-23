from copy import deepcopy
import pickle
import graphviz as gv
from IPython.display import Image
from IPython.display import display
import functools
import string

digraph = functools.partial(gv.Digraph, format='png')
graph = functools.partial(gv.Graph, format='png')

separator = "_"

class Moore():
    def __init__(self,obs_table):
        if obs_table is not None:
            self.alphabet = obs_table.A
            self.Q = [s for s in obs_table.S if s==obs_table.minimum_matching_row(s)]
            self.q0    = obs_table.minimum_matching_row("")
            self.delta = self._make_transition_function(obs_table)
            self.X     = self._make_output_function(obs_table)
    
    def initialize_static(alphabet, Q, q0, delta, X):
        dfa = Moore(None)

        dfa.alphabet = alphabet
        dfa.Q = Q
        dfa.q0 = q0
        dfa.delta = delta   
        dfa.X = X

        return dfa
    
    def _make_transition_function(self, obs_table):
        delta = {}
        for s in self.Q:
            delta[s] = {}
            for a in self.alphabet:
                delta[s][a] = obs_table.minimum_matching_row(s+a)
        return delta

    def _make_output_function(self, obs_table):
        X = {self.q0 : obs_table.T[self.q0]}
        for s in self.Q:
            for a in self.alphabet:
                if self.delta[s][a] in X:
                    assert X[self.delta[s][a]] == obs_table.T[s+a], "This observation table does not contain a Moore machine!"
                else:
                    X[self.delta[s][a]] = obs_table.T[s+a] 
        return X
    
    def draw_nicely(self,force=False,maximum=60, name=None): #todo: if two edges are identical except for letter, merge them and note both the letters
        if (not force) and len(self.Q) > maximum:
            return

        def label_to_numberlabel(label):
            return str(self.Q.index(label))

        def add_nodes(graph, nodes):
            for n in nodes:
                if isinstance(n, tuple):
                    graph.node(n[0], **n[1])
                else:
                    graph.node(n)
            return graph

        def add_edges(graph, edges):
            for e in edges:
                if isinstance(e[0], tuple):
                    graph.edge(*e[0], **e[1])
                else:
                    graph.edge(*e)
            return graph

        g = digraph()
        g = add_nodes(g, [(label_to_numberlabel(self.q0), {'color':'black',
                                     'shape': 'hexagon', 'label':str(self.X[self.q0])})])
        states = list(set(self.Q)-{self.q0})
        g = add_nodes(g, [(label_to_numberlabel(state),{'color': 'black',
                                  'label': str(self.X[state])})
                          for state,i in zip(states,range(1,len(states)+1))])

        def group_edges():
            def clean_line(line,group):
                line = line.split(separator)
                line = sorted(line) + ["END"]
                in_sequence= False
                last_a = ""
                clean = line[0]
                if line[0] in group:
                    in_sequence = True
                    first_a = line[0]
                    last_a = line[0]
                for a in line[1:]:
                    if in_sequence:
                        if a in group and (ord(a)-ord(last_a))==1: #continue sequence
                            last_a = a
                        else: #break sequence
                            #finish sequence that was
                            if (ord(last_a)-ord(first_a))>1:
                                clean += ("-" + last_a)
                            elif not last_a == first_a:
                                clean += (separator + last_a)
                            #else: last_a==first_a -- nothing to add
                            in_sequence = False
                            #check if there is a new one
                            if a in group:
                                first_a = a
                                last_a = a
                                in_sequence = True
                            if not a=="END":
                                clean += (separator + a)
                    else:
                        if a in group: #start sequence
                            first_a = a
                            last_a = a
                            in_sequence = True
                        if not a=="END":
                            clean += (separator+a)
                return clean


            edges_dict = {}
            for state in self.Q:
                for a in self.alphabet:
                    edge_tuple = (label_to_numberlabel(state),label_to_numberlabel(self.delta[state][a]))
                    if not edge_tuple in edges_dict:
                        edges_dict[edge_tuple] = a
                    else:
                        edges_dict[edge_tuple] += separator+a
            for et in edges_dict:
                edges_dict[et] = clean_line(edges_dict[et], string.ascii_lowercase)
                edges_dict[et] = clean_line(edges_dict[et], string.ascii_uppercase)
                edges_dict[et] = clean_line(edges_dict[et], "0123456789")
                edges_dict[et] = edges_dict[et].replace(separator,",")
            return edges_dict

        edges_dict = group_edges()
        g = add_edges(g,[(e,{'label': edges_dict[e],"color": "black"}) for e in edges_dict])
        if name is None:
            display(Image(filename=g.render(filename='img/automaton')))
        else:
            display(Image(filename=g.render(filename=f'img/{name}')))

    """
    Use breadth first search, no duplicates, until max_depth
    """
    def get_returning_suffixes(self, start_state, max_depth=5, optimize_garbage_state=False, garbage_state=None, max_suffixes=None):
        curr_states = {start_state: [""]}
        returning_suffixes = []

        for d in range(max_depth):
            print(f"searching depth {d}", end="\r")

            next_states = {}
            for state in curr_states.keys():
                for a in self.alphabet:
                    next_state = self.delta[state][a]

                    prev_suffixes = next_states.get(next_state)
                    curr_suffixes = curr_states[state]
                    new_suffixes = [suffix + a for suffix in curr_suffixes]
                    if prev_suffixes is None:
                        next_states.update({next_state: new_suffixes})
                    else:
                        next_states.update({next_state: prev_suffixes + new_suffixes})

        
            if max_suffixes is not None:
                for state in next_states.keys():
                    if len(next_states[state]) > max_suffixes:
                        next_states[state] = np.random.choice(next_states[state], max_suffixes, replace=False).tolist()
            curr_returning_suffixes = next_states.get(start_state)
            if curr_returning_suffixes is not None:
                returning_suffixes += curr_returning_suffixes
            if optimize_garbage_state and next_states.get(garbage_state) is not None:
                next_states.pop(garbage_state)
            curr_states = next_states
        
        return returning_suffixes
        
    """
    All ways to reach a state from the start state
    Up to max_length and without passing through the target state before
    """
    def get_arriving_prefixes(self, target_state, max_length, optimize_garbage_state=False, garbage_states=[]):
        arriving_prefixes = []
        curr_states_prefixes = {(self.q0, "")}

        for _ in range(max_length):
            next_states_prefixes = []
            for curr_state, curr_prefix in curr_states_prefixes:
                for a in self.alphabet:
                    next_state = self.delta[curr_state][a]
                    next_prefix = curr_prefix + a
                    if next_state == target_state:
                        arriving_prefixes.append(next_prefix)
                    elif optimize_garbage_state and next_state in garbage_states:
                        pass
                    else:
                        next_states_prefixes.append((next_state, next_prefix))
            curr_states_prefixes = next_states_prefixes

        return arriving_prefixes
        
    """
    Get word return output of last transition taken
    """
    def classify_word(self,word):
        #assumes word is string with only letters in alphabet
        #support alphabet with multicharacter strings
        q = self.q0
        next_word = ""
        for a in word:
            next_word += a
            if next_word in self.alphabet:
                q = self.delta[q][next_word]
                next_word = ""
            
        return self.X[q]
    
    def minimal_diverging_suffix(self,state1,state2): 
        #gets series of letters showing the two states are different,
        # i.e., from which one state reaches accepting state and the other reaches rejecting state
        # assumes of course that the states are in the automaton and actually not equivalent
        res = None
        # just use BFS until you reach an accepting state

        seen_states = set()
        new_states = {("",(state1,state2))}
        while len(new_states) > 0:
            prefix,state_pair = new_states.pop()
            s1,s2 = state_pair
            if self.X[s1] != self.X[s2]:
                # meaning s1 and s2 are classified differently
                res = prefix
                break
            seen_states.add(state_pair)
            for a in self.alphabet:
                q1 = self.delta[s1][a]
                q2 = self.delta[s2][a]
                next_state_pair = (q1, q2)
                next_tuple = (prefix+a,next_state_pair)
                if not next_tuple in new_states and not next_state_pair in seen_states:
                    new_states.add(next_tuple)
        return res

    """
    Return list of labels for each character in the word
    """
    def label_word(self, word):
        curr_q = self.q0
        labels = []
        for c in word:
            curr_q = self.delta[curr_q][c]
            labels.append(self.X[curr_q])
        
        return labels

    """
    file_name: just name - no extension
    """
    def save(self, file_path):
        with open("automata/"+file_path+".pkl", mode="wb") as f:
            pickle.dump(self, f)

    def load(file_name):
        with open("automata/"+file_name+".pkl", mode="rb") as f:
            dfa = pickle.load(f)

        return dfa
    
    def copy(self):
        return Moore.initialize_static(
            alphabet=self.alphabet.copy(), 
            Q=self.Q.copy(), 
            q0=self.q0, 
            delta=deepcopy(self.delta),
            X=deepcopy(self.X))