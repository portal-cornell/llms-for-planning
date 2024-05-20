"""
This module contains utility functions that can be shared across different policies.
"""
import string
import ast
import re
from copy import deepcopy

def get_actions_to_propose_cheap(graph, model, state):
    """Returns the actions to propose to reach the goal.

    This performs a set difference between the valid actions and the actions already taken in the graph
    to ensure that the same action is not proposed twice; however, actions may lead to states that have
    already been visited.

    Parameters:
        graph (nx.DiGraph)
            The graph to propose actions in.
        model (Model)
            The model to propose actions with.
        state (object)
            The current state of the environment.
    
    Returns:
        actions_to_propose (list)
            The actions to propose to reach the goal.
    """
    valid_actions = model.get_valid_actions(state)
    actions_taken = [graph[hash(state)][node]["action"] for node in graph.successors(hash(state))]
    return list(set(valid_actions) - set(actions_taken))

def get_actions_to_propose(graph, model, state):
    """Returns the actions to propose to reach the goal.

    This function is similar to get_actions_to_propose_cheap, but it checks if the next state has already
    been visited in the graph before proposing the action.

    Parameters:
        graph (nx.DiGraph)
            The graph to propose actions in.
        model (Model)
            The model to propose actions with.
        state (object)
            The current state of the environment.
    
    Returns:
        actions_to_propose (list)
            The actions to propose to reach the goal.
    """
    valid_actions = model.get_valid_actions(state)
    actions_to_propose = []
    for action in valid_actions:
        model_copy = deepcopy(model)
        next_state, _, _, _, _ = model_copy.env.step(action)
        if hash(next_state) not in graph.nodes:
            actions_to_propose.append(action)
    return actions_to_propose

def get_box_info(literal_str):
    box, coords = literal_str.split(',')
    return int(box[box.index('box') + 3 : box.rindex(':box')]), int(coords[1 : coords.index('-')]), int(coords[coords.index('-') + 1 : coords.rindex('f')])


def convert_states_to_bitmap_sokoban(all_states, is_goal=False):
    box_letters = string.ascii_lowercase
    if is_goal:
        goals = []
        for literal in all_states.literals:
            box, r, c = get_box_info(str(literal))
            goals.append(f'box {box_letters[box]}: {(r,c)}')
        return '\n'.join(goals)

    max_grid_ind = 4
    regex = 'adjacent\(f([0-9]*)-([0-9]*)f:loc,f([0-9]*)-([0-9]*)f:loc,([a-z]*):dir\)'
    for literal in all_states.literals:
        literal_str = str(literal)
        if literal_str.startswith('adjacent'):
            m = re.search(regex, literal_str)
            max_grid_ind = max([int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), max_grid_ind])
    
    robot_coord, walls, boxes, goals = None, [], {}, {}
    grid_size = max_grid_ind + 1
    grid = [[1] * grid_size for _ in range(grid_size)]
    for literal in all_states.literals:
        literal_str = str(literal)
        if literal_str.startswith('clear'):
            r, c = literal_str.split('-')
            r, c = int(r[r.index('f') + 1 : ]), int(c[ : c.index('f')])
            if grid[r][c] == 1:
                grid[r][c] = 0 # grid without wall
        elif literal_str.startswith('at('):
            box, r, c = get_box_info(literal_str)
            grid[r][c] = box_letters[box] # box starting location
            boxes[box_letters[box]] = (r, c)
        elif literal_str.startswith('at-robot'):
            _, r, c, = literal_str.split('-')
            r, c = int(r[r.index('f') + 1 :]), int(c[ : c.index('f')])
            grid[r][c] = 'r' # robot starting location
            robot_coord = (r, c)
    
    for literal in all_states.goal.literals:
        literal_str = str(literal)
        box, r, c = get_box_info(literal_str)
        grid[r][c] = box_letters[box].upper() # box goal location
        goals[box_letters[box]] = (r, c)
        
    
    s = ''
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            s += str(cell)
            if cell == 1:
                walls.append((r, c))
        s += '\n'
    
    s += f"\nObject locations:\nrobot: {robot_coord}\nwalls: {', '.join([str(s) for s in walls])}\nboxes: {', '.join([letter + ': ' + str(coord) for letter, coord in boxes.items()])}\ngoals: {', '.join([letter + ': ' + str(coord) for letter, coord in goals.items()])}"
    
    return s + "\n"

def map_llm_action_sokoban(action_str):
    def filter_diff(start_r, end_r, start_c, end_c):
        diffs = [[(1, 0), "down"], [(0, 1), "right"], [(-1, 0), "up"], [(0, -1), "down"]]
        diff = (end_r - start_r, end_c - start_c)
        filtered_diff = list(filter(lambda tup : tup[0] == diff, diffs))

        return filtered_diff

    parsed_action = ''
    if action_str.startswith('move'):
        (start_r, start_c), (end_r, end_c) = ast.literal_eval(action_str[4:])
        filtered_diff = filter_diff(start_r, end_r, start_c, end_c)
        if len(filtered_diff):
            _, direction = filtered_diff[0]
            parsed_action = f'move(f{start_r}-{start_c}f:loc,f{end_r}-{end_c}f:loc,{direction}:dir)'
    elif action_str.startswith('push'):
        (start_r, start_c), (end_r, end_c), _ = ast.literal_eval(action_str[4:])
        filtered_diff = filter_diff(start_r, end_r, start_c, end_c)
        if len(filtered_diff):
            diff, direction = filtered_diff[0]
            block_end_r, block_end_c = end_r + diff[0], end_c + diff[1]
            parsed_action = f'push(f{start_r}-{start_c}f:loc,f{end_r}-{end_c}f:loc,f{block_end_r}-{block_end_c}f:loc,{direction}:dir)'
    
    return parsed_action

def convert_game_states_to_bitmap(state):
    max_grid_dim = 2
    tile_letters = string.ascii_lowercase
    tile_coords = {}
    for literal in state.literals:
        literal_str = str(literal)
        if literal_str.startswith('neighbor'):
            regex = 'neighbor\(p_([0-9])_([0-9]):position,p_([0-9])_([0-9]):position\)'
            m = re.search(regex, literal_str)
            r_1, c_1, r_2, c_2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            max_grid_dim = max([max_grid_dim, r_1, c_1, r_2, c_2])
        elif literal_str.startswith('at'):
            regex = 'at\(t_([0-9]):tile,p_([0-9])_([0-9]):position\)'
            m = re.search(regex, literal_str)
            tile_ind, r, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
            max_grid_dim = max([max_grid_dim, r, c])
            tile_coords[tile_letters[tile_ind]] = (r - 1, c - 1)
    
    bitmap = [['_'] * max_grid_dim for _ in range(max_grid_dim)]
    for tile_letter, (r, c) in tile_coords.items():
        bitmap[r][c] = tile_letter
    
    s = 'Bitmap representation:\n'
    for r, row in enumerate(bitmap):
        for c, cell in enumerate(row):
            s += str(cell)
        s += '\n'
    s += '\nCoordinate representation:\n'
    for tile_letter, tup in tile_coords.items():
        s += f'{tile_letter}: {tup}\n'

    return s

def pretty_pddl_state(state, domain, model, is_goal=False):
    if domain == 'logistics':
        if not is_goal:
            state_str = model.state_to_str(state)
        else:
            state_str = model.goal_to_str([], state)
        return state_str.replace(':default', '')
    elif domain == 'grippers':
        if not is_goal:
            state_str = model.state_to_str(state)
        else:
            state_str = model.goal_to_str([], state)
        for s in ['robot', 'object', 'gripper', 'room']:
            state_str = state_str.replace(f':{s}', '')
        return state_str
    elif domain == 'hanoi':
        literals = [str(literal) for literal in state.literals if not 'smaller(peg' in str(literal)]
        if not is_goal:
            objects = [str(obj) for obj in state.objects]
            str_state = f"""Predicates: {', '.join(literals)}
            Objects: {', '.join(objects)}"""
            return str_state.replace(':default', '')

        return f"Goal: {', '.join(literals)}"
    elif domain == 'sokoban':
        return convert_states_to_bitmap_sokoban(state, is_goal)
    elif domain == 'game':
        return convert_game_states_to_bitmap(state)
    else:
        if not is_goal:
            state_str = model.state_to_str(state)
        else:
            state_str = model.goal_to_str([], state)

def translate_literal(literal, domain):
    if domain in ['logistics', 'hanoi']:
        return literal.replace(':default', '')
    elif domain == 'gippers':
        for s in ['robot', 'object', 'gripper', 'room']:
            literal = literal.replace(f':{s}', '')
        return literal
    elif domain == 'sokoban':
        box, r, c = get_box_info(str(literal))
        return f'box {string.ascii_lowercase[box]} at {(r,c)}'
    elif domain == 'game':
        tile_letters = string.ascii_lowercase
        regex = 'at\(t_([0-9]):tile,p_([0-9])_([0-9]):position\)'
        m = re.search(regex, literal)
        tile_ind, r, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f'tile {tile_letters[tile_ind]} at {(r-1, c-1)}'
    else:
        return literal

def pretty_pddl_actions(action_str, domain):
    if domain == 'logistics':
        return action_str.replace(':default', '')
    elif domain == 'grippers':
        for s in ['robot', 'object', 'gripper', 'room']:
            action_str = action_str.replace(f':{s}', '')
        return action_str
    elif domain == "hanoi":
        return action_str.replace(':default', '')
    elif domain == 'sokoban':
        if action_str.startswith('move'):
            regex = 'move\(f([0-9]*)-([0-9]*)f:loc,f([0-9]*)-([0-9]*)f:loc,([a-z]*):dir\)'
            m = re.search(regex, action_str)
            start_r, start_c, end_r, end_c = m.group(1), m.group(2), m.group(3), m.group(4)
            return f"move(({start_r}, {start_c}), ({end_r}, {end_c}))"
        else:
            regex = 'push\(f([0-9]*)-([0-9]*)f:loc,f([0-9]*)-([0-9]*)f:loc,f([0-9]*)-([0-9]*)f:loc,([a-z]*):dir,box[0-9]*:box\)'
            m = re.search(regex, action_str)
            start_r, start_c, end_r, end_c = m.group(1), m.group(2), m.group(3), m.group(4)
            return f"push(({start_r}, {start_c}), ({end_r}, {end_c}))"
    elif domain == 'game':
        regex = 'move\(t_([0-9]):tile,p_([0-9])_([0-9]):position,p_([0-9])_([0-9]):position\)'
        m = re.search(regex, action_str)
        tire_letter_ind, start_r, start_c, end_r, end_c = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        return f'move({string.ascii_lowercase[tire_letter_ind]},({start_r-1},{start_c-1}),({end_r-1},{end_c-1}))'
    else:
        return action_str