#travelling salesman
from sys import maxsize
from itertools import permutations

V = 4
def travellingSalesmanProblem(graph, s):
    vertex = []
    for i in range(V):
        if i != s:
            vertex.append(i)

    min_path = maxsize
    for i in permutations(vertex):
        current_pathweight = 0
        k = s
        for j in i:
            current_pathweight += graph[k][j]
            k = j
        current_pathweight += graph[k][s]
        min_path = min(min_path, current_pathweight)

    return min_path

if __name__ == "__main__":
    graph = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    s = 0
    print(travellingSalesmanProblem(graph, s))

# graph coloring 
colors = ['Red', 'Blue', 'Green', 'Yellow', 'Black']
states = ['Telangana', 'Karnataka', 'TamilNadu', 'Kerala']

neighbors = {
    'Telangana': ['Karnataka', 'TamilNadu'],
    'Karnataka': ['Telangana', 'TamilNadu', 'Kerala'],
    'TamilNadu': ['Telangana', 'Karnataka', 'Kerala'],
    'Kerala': ['Karnataka', 'TamilNadu']
}

colors_of_states = {}

def promising(state, color):
    for neighbor in neighbors.get(state):
        if colors_of_states.get(neighbor) == color:
            return False
    return True

def get_color_for_state(state):
    for color in colors:
        if promising(state, color):
            return color

def main():
    for state in states:
        colors_of_states[state] = get_color_for_state(state)
    print(colors_of_states)

main()
