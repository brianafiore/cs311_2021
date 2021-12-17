import random
import sys

EMPTY = ' '
WALL = '#'
AGENT = 'o'
GOAL = 'x'


def adjacent(cell):
    i, j = cell
    for (y, x) in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        yield (i + y, j + x), (i + 2 * y, j + 2 * x)


def generate(width, height, verbose=True):
    '''Generates a maze as a list of strings.
       :param width: the width of the maze, not including border walls.
       :param heihgt: height of the maze, not including border walls.
    '''
    # add 2 for border walls.
    width += 2
    height += 2
    rows, cols = height, width

    maze = {}

    spaceCells = set()
    connected = set()
    walls = set()

    # Initialize with grid.
    for i in range(rows):
        for j in range(cols):
            if (i % 2 == 1) and (j % 2 == 1):
                maze[(i, j)] = EMPTY
            else:
                maze[(i, j)] = WALL

    # Fill in border.
    for i in range(rows):
        maze[(i, 0)] = WALL
        maze[(i, cols - 1)] = WALL
    for j in range(cols):
        maze[(0, j)] = WALL
        maze[(rows - 1, j)] = WALL

    for i in range(rows):
        for j in range(cols):
            if maze[(i, j)] == EMPTY:
                spaceCells.add((i, j))
            if maze[(i, j)] == WALL:
                walls.add((i, j))

    # Prim's algorithm to knock down walls.
    originalSize = len(spaceCells)
    connected.add((1, 1))
    while len(connected) < len(spaceCells):
        doA, doB = None, None
        cns = list(connected)
        random.shuffle(cns)
        for (i, j) in cns:
            if doA is not None: break
            for A, B in adjacent((i, j)):
                if A not in walls:
                    continue
                if (B not in spaceCells) or (B in connected):
                    continue
                doA, doB = A, B
                break
        A, B = doA, doB
        maze[A] = EMPTY
        walls.remove(A)
        spaceCells.add(A)
        connected.add(A)
        connected.add(B)
        if verbose:
            cs, ss = len(connected), len(spaceCells)
            cs += (originalSize - ss)
            ss += (originalSize - ss)

    # Insert character and goals.
    TL = (1, 1)
    BR = (rows - 2, cols - 2)
    if rows % 2 == 0:
        BR = (BR[0] - 1, BR[1])
    if cols % 2 == 0:
        BR = (BR[0], BR[1] - 1)

    maze[TL] = AGENT
    maze[BR] = GOAL

    lines = []
    for i in range(rows):
        lines.append(''.join(maze[(i, j)] for j in range(cols)))

    return lines


# CONVERT TO LIST
def convert_to_matrix(maze):
    # maze_string = maze.read
    adj_matrix = []
    lines = maze.splitlines()
    for char in lines:
        adj_matrix.append(list(char))
    return adj_matrix


def convert_to_int(matrix):
    for i in range(len(matrix)):
        matrix[i] = list(map(lambda x: int(x != '#'), matrix[i]))
    return matrix


def get_agent(maze):
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if maze[y][x] == AGENT:
                agent = y, x
                return agent


def get_goal(maze):
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if maze[y][x] == GOAL:
                goal = y, x
                return goal


def astar(maze, start, end) -> object:
    start = Node(None, start)
    start.g = start.h = start.f = 0
    goal = Node(None, end)
    goal.g = goal.h = goal.f = 0

    empty_cells = []
    walls = []

    empty_cells.append(start)

    while len(empty_cells) > 0:

        # CURRENT POSITION
        current_cell = empty_cells[0]
        current_coord = 0
        for coord, current_cell in enumerate(empty_cells):
            if current_cell.f < current_cell.f:
                current_cell = current_cell
                current_coord = coord

        # ADD TO WALLS LIST
        empty_cells.pop(current_coord)
        walls.append(current_cell)

        # COMPLETE MAZE
        if current_cell == goal:
            path = []
            current = current_cell
            while current is not None:
                path.append(current.position)
                current = current.parent

            return path[::-1]  # Return reversed path
            # print(path[::-1])

        # MAKE NEIGHBOR LIST
        neighbors = []
        # ASSIGN ADJACENT MOVE RESTRICTIONS
        for new_position in [(0, -1),  # LEFT
                             (0, 1),  # RIGHT
                             (-1, 0),  # UP
                             (1, 0)]:  # DOWN

            # CURRENT POSITION
            cell_position = (current_cell.position[0] + new_position[0], current_cell.position[1] + new_position[1])

            # VERIFY LOCATION
            if cell_position[0] > (len(maze) - 1) or cell_position[0] < 0 or cell_position[1] > (
                    len(maze[len(maze) - 1]) - 1) or cell_position[1] < 0:
                continue

            # VALIDATE POSITION
            if maze[cell_position[0]][cell_position[1]] != 0:
                continue

            # ADD NODE
            new_cell = Node(current_cell, cell_position)

            # ADD NEIGHBORS
            neighbors.append(new_cell)

        for cell in neighbors:
            # IF VISITED
            for closed_child in walls:
                if cell == closed_child:
                    continue

            # SET VARIABLES FOR A* FORMULA
            cell.g = current_cell.g + 1
            cell.h = ((cell.position[0] - goal.position[0]) ** 2) + (
                    (cell.position[1] - goal.position[1]) ** 2)
            cell.f = cell.g + cell.h

            # VALIDATE CELL
            for index in empty_cells:
                if cell == index and cell.g > index.g:
                    continue

            # ADD TO EMPTY LIST
            empty_cells.append(cell)

    return


class Node():
    def __init__(self, parent_cell=None, position=None):
        self.parent = parent_cell
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


# *********************

if __name__ == '__main__':
    width = 9
    height = 9

    args = sys.argv[1:]
    if len(args) >= 1:
        width = int(args[0])
    if len(args) >= 2:
        height = int(args[1])

    maze = generate(width, height)
    maze_string = str('\n'.join(maze))

    print(maze_string)

    print('\nCONVERT STRING TO MATRIX: ')
    maze_matrix = convert_to_matrix(maze_string)
    print(maze_matrix)
    start = get_agent(maze_matrix)
    start_obj = Node(get_agent(maze_matrix))
    print('START: ', start)
    goal = get_goal(maze_matrix)
    goal_obj = Node(get_goal(maze_matrix))
    print('END: ', goal)

    int_matrix = [[1 if cell != WALL else 0 for cell in m] for m in maze_matrix]
    # int_matrix = ((1 if cell != WALL else 0 for cell in m) for m in maze_matrix)

    print(int_matrix)


    path = astar(int_matrix, start_obj, goal_obj)
    print(path)
