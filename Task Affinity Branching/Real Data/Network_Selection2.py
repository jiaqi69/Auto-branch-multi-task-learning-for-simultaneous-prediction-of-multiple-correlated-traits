
# coding: utf-8

import math
import numpy as np
import os
import ast
import time

save_path = 'D:/gains/'
for i in range(1, 101):
    filename = f'{save_path}gains_{i}.txt'

    with open(filename, 'r') as file:
        lines = file.readlines()
    lines = lines[:223]

    with open(filename, 'w') as file:
        file.writelines(lines)


def calculate_average_affinity(file_name):
    with open(file_name, 'r') as file:
        data = file.read()

    task_data = data.split('\n\n')
    revised_integrals = {}

    for task in task_data:
        lines = task.strip().split('\n')
        task_name = lines[0].rstrip(':')
        task_affinity = {}

        for line in lines[1:]:
            affinity_dict = ast.literal_eval(line)

            for key, value in affinity_dict.items():
                if key not in task_affinity:
                    task_affinity[key] = []

                task_affinity[key].append(value)

        task_average = {key: sum(values) / len(values) for key, values in task_affinity.items()}

        revised_integrals[task_name] = task_average

    return revised_integrals


# ### Network Selection
result_file_path = os.path.join(save_path, "group_result_3.txt")

start_time = time.time()
for i in range(1, 101):
    print(i)
    filename = f'gains_{i}.txt'
    revised_integrals = calculate_average_affinity(os.path.join(save_path, filename))
    
    def gen_task_combinations(tasks, rtn, index, path, path_dict):
        if index >= len(tasks):
            return 
    
        for i in range(index, len(tasks)):
            cur_task = tasks[i]
            new_path = path
            new_dict = {k:v for k,v in path_dict.items()}
            
            # Building from a tree with two or more tasks...
            if new_path:
                new_dict[cur_task] = 0.
                for prev_task in path_dict:
                    new_dict[prev_task] += revised_integrals[prev_task][cur_task]
                    new_dict[cur_task] += revised_integrals[cur_task][prev_task]
                new_path = '{}|{}'.format(new_path, cur_task)
                rtn[new_path] = new_dict
            else: # First element in a new-formed tree
                new_dict[cur_task] = 0.
                new_path = cur_task
    
            gen_task_combinations(tasks, rtn, i+1, new_path, new_dict)
    
    
            if '|' not in new_path:
                new_dict[cur_task] = -1e6 
                rtn[new_path] = new_dict
        
    rtn = {}
    tasks = list(revised_integrals.keys())
    num_tasks = len(tasks)
    task_combinations = gen_task_combinations(tasks=tasks, rtn=rtn, index=0, path='', path_dict={})
    
    # Normalize by the number of times the accuracy of any given element has been summed. 
    # i.e. (a,b,c) => [acc(a|b) + acc(a|c)]/2 + [acc(b|a) + acc(b|c)]/2 + [acc(c|a) + acc(c|b)]/2
    for group in rtn:
        if '|' in group:
            for task in rtn[group]:
                rtn[group][task] /= (len(group.split('|')) - 1)
    
    #print(rtn)
    assert(len(rtn.keys()) == 2**len(revised_integrals.keys()) - 1)
    rtn_tup = [(key,val) for key,val in rtn.items()]
    
    
    
    def select_groups(index, cur_group, best_group, best_val, splits):
        # Check if this group covers all tasks.
        task_set = set()
        for group in cur_group:
            for task in group.split('|'): task_set.add(task)
        if len(task_set) == num_tasks:
            best_tasks = {task:-1e6 for task in task_set}
          
          # Compute the per-task best scores for each task and average them together.
            for group in cur_group:
                for task in cur_group[group]:
                    best_tasks[task] = max(best_tasks[task], cur_group[group][task])
            group_avg = np.mean(list(best_tasks.values()))
            
            # Compare with the best grouping seen thus far.
            if group_avg > best_val[0]:
                #print(cur_group)
                best_val[0] = group_avg
                best_group.clear()
                for entry in cur_group:
                    best_group[entry] = cur_group[entry]
        
        # Base case.
        if len(cur_group.keys()) == splits:
            return
    
        # Back to combinatorics 
        for i in range(index, len(rtn_tup)):
            selected_group, selected_dict = rtn_tup[i]
    
            new_group = {k:v for k,v in cur_group.items()}
            new_group[selected_group] = selected_dict
    
            if len(new_group.keys()) <= splits:
                select_groups(i + 1, new_group, best_group, best_val, splits)
    
    selected_group = {}
    selected_val = [-100000000]
    select_groups(index=0, cur_group={}, best_group=selected_group, best_val=selected_val, splits=3)
    #print(selected_group)
    #print(selected_val)


    task_max_values = {}
    for group, tasks in selected_group.items():
        for task, value in tasks.items():
            if task not in task_max_values:
                task_max_values[task] = {'group': group, 'value': value}
            else:
                if value > task_max_values[task]['value']:
                    task_max_values[task] = {'group': group, 'value': value}

    final_selected_group = {}
    for task, info in task_max_values.items():
        final_selected_group.setdefault(info['group'], {})[task] = info['value']

    for group, tasks in selected_group.items():
        for task, value in tasks.items():
            if task not in task_max_values:
                final_selected_group.setdefault(group, {})[task] = value

    values_list = list(final_selected_group.values())
    
    from collections import OrderedDict
    keys_set = [list(OrderedDict.fromkeys(dictionary)) for dictionary in values_list]
    
    with open(result_file_path, 'a') as result_file:
        result_file.write(f"{keys_set}\n")

print("Completed!")
end_time = time.time() 
elapsed_time = end_time - start_time
print(f"Total time{elapsed_time}")





