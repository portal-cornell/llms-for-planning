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

def get_actions_to_propose_cheap_robotouille(graph, model, state):
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
    valid_actions, _ = model.get_valid_actions(model.env.current_state)
    hasher = lambda va: [str(valid_actions) for valid_actions in va]
    actions_taken = [graph[hash(state)][node]["action"] for node in graph.successors(hash(state))]
    hashable_valid_actions = hasher(valid_actions)
    hashable_actions_taken = hasher(actions_taken)
    set_diff = set(hashable_valid_actions) - set(hashable_actions_taken)
    # Get original valid actions
    valid_actions = [va for va in valid_actions if str(va) in set_diff]
    return valid_actions

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

def convert_states_to_bitmap_sokoban(all_states):
    def get_box_info(literal_str):
        box, coords = literal_str.split(',')
        return int(box[box.index('box') + 3 : box.rindex(':box')]), int(coords[1 : coords.index('-')]), int(coords[coords.index('-') + 1 : coords.rindex('f')])

    max_grid_ind = 4
    for state in all_states.objects:
        state_str = str(state)
        if state_str.startswith('f'):
            first, _ = state_str.split('-')
            max_grid_ind = max(max_grid_ind, int(first[1]))
    
    grid_size = max_grid_ind + 1
    grid = [[1] * grid_size for _ in range(grid_size)]
    box_letters = string.ascii_lowercase
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
        elif literal_str.startswith('at-robot'):
            _, r, c, = literal_str.split('-')
            r, c = int(r[r.index('f') + 1 :]), int(c[ : c.index('f')])
            grid[r][c] = 'r' # robot starting location
    
    for literal in all_states.goal.literals:
        literal_str = str(literal)
        box, r, c = get_box_info(literal_str)
        grid[r][c] = box_letters[box].upper() # box goal location
    
    s = ''
    for row in grid:
        for cell in row:
            s += str(cell)
        s += '\n'
    
    return s.strip()

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

def convert_logistics_states(state, model, is_goal=False):
    if not is_goal:
        state_str = model.state_to_str(state)
    else:
        state_str = model.goal_to_str([], state)
    state_str = state_str.replace(':default', '')

    at_regex = 'at\(([atp])([0-9]+),l([0-9]+)-([0-9]+)\)'
    in_regex = 'in\(p([0-9]+),([ta])([0-9]+)\)'
    airport_regex = 'airport\(l([0-9]+)-([0-9]+)\)'
    
    processed_predicates = []
    for m in re.findall(airport_regex, state_str):
        loc_1, loc_2 = m
        pred = f'there is an airport at location {loc_1}-{loc_2}'
        processed_predicates.append(pred)

    for m in re.findall(at_regex, state_str):
        obj_type, obj_num, loc_1, loc_2 = m
        loc_pred = f'location {loc_1}-{loc_2} is in city {loc_1}'
        obj = {'a': 'airplane', 't': 'truck', 'p': 'package'}[obj_type]
        obj_pred = f'{obj} {obj_num} is in location {loc_1}-{loc_2}'
        for pred in [loc_pred, obj_pred]:
            if not pred in processed_predicates:
                processed_predicates.append(pred)
    
    for m in re.findall(in_regex, state_str):
        package_num, obj_type, obj_num = m
        obj = {'a': 'airplane', 't': 'truck', 'p': 'package'}[obj_type]
        pred = f'package {package_num} is in {obj} {obj_num}'
        if not pred in processed_predicates:
            processed_predicates.append(pred)
    
    return '\n'.join(processed_predicates)

def convert_grippers_states(state, model, is_goal=False):
    if not is_goal:
        state_str = model.state_to_str(state)
    else:
        state_str = model.goal_to_str([], state)
    for s in ['robot', 'object', 'gripper', 'room']:
        state_str = state_str.replace(f':{s}', '')
    return state_str

def pretty_pddl_state(state, model, is_goal, domain="blocksworld"):
    if domain == 'logistics':
        return convert_logistics_states(state, model, is_goal)
    elif domain == 'grippers':
        return convert_grippers_states(state, model, is_goal)
    else:
        if not is_goal:
            state_str = model.state_to_str(state)
        else:
            state_str = model.goal_to_str([], state)
        return state_str

def translate_logistics_actions(action_str):
    action_str = action_str.replace(':default', '')
    load_truck_regex = 'load-truck\(p([0-9]+),t([0-9]+),l([0-9]+)-([0-9]+)\)'
    load_airplane_regex = 'load-airplane\(p([0-9]+),a([0-9]+),l([0-9]+)-([0-9]+)\)'
    unload_truck_regex = 'unload-truck\(p([0-9]+),t([0-9]+),l([0-9]+)-([0-9]+)\)'
    unload_airplane_regex = 'unload-airplane\(p([0-9]+),a([0-9]+),l([0-9]+)-([0-9]+)\)'
    drive_truck_regex = 'drive-truck\(t([0-9]+),l([0-9]+)-([0-9]+),l([0-9]+)-([0-9]+),c([0-9]+)\)'
    fly_airplane_regex = 'fly-airplane\(a([0-9]+),l([0-9]+)-([0-9]+),l([0-9]+)-([0-9]+)\)'

    if action_str.startswith('load-truck'):
        m = re.search(load_truck_regex, action_str)
        package_num, truck_num, city_num, loc_num = [m.group(i) for i in range(1,5)]
        return f'load package {package_num} onto truck {truck_num} at location {city_num}-{loc_num}'
    elif action_str.startswith('load-airplane'):
        m = re.search(load_airplane_regex, action_str)
        package_num, plane_num, city_num, loc_num = [m.group(i) for i in range(1,5)]
        return f'load package {package_num} onto airplane {plane_num} at location {city_num}-{loc_num}'
    elif action_str.startswith('unload-truck'):
        m = re.search(unload_truck_regex, action_str)
        package_num, truck_num, city_num, loc_num = [m.group(i) for i in range(1,5)]
        return f'unload package {package_num} from truck {truck_num} at location {city_num}-{loc_num}'
    elif action_str.startswith('unload-airplane'):
        m = re.search(unload_airplane_regex, action_str)
        package_num, plane_num, city_num, loc_num = [m.group(i) for i in range(1,5)]
        return f'unload package {package_num} from airplane {plane_num} at location {city_num}-{loc_num}'
    elif action_str.startswith('drive-truck'):
        m = re.search(drive_truck_regex, action_str)
        truck_num, loc_11, loc_12, loc_21, loc_22, city_num = [m.group(i) for i in range(1, 7)]
        return f'drive truck {truck_num} from location {loc_11}-{loc_12} to location {loc_21}-{loc_22} within city {city_num}'
    else:
        m = re.search(fly_airplane_regex, action_str)
        airplane_num, loc_11, loc_12, loc_21, loc_22 = [m.group(i) for i in range(1, 6)]
        return f'fly airplane {airplane_num} from the airport at location {loc_11}-{loc_12} in city {loc_11} to the airport at location {loc_21}-{loc_22} in city {loc_21}'

def translate_grippers_actions(action_str):
    for s in ['robot', 'object', 'gripper', 'room']:
        action_str = action_str.replace(f':{s}', '')
    return action_str

def pretty_pddl_actions(action_str, domain):
    if domain == 'logistics':
        return translate_logistics_actions(action_str)
    elif domain == 'grippers':
        return translate_grippers_actions(action_str)
    else:
        return action_str