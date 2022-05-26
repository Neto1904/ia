import numpy as np

class Fuzzy:

    def __init__(self):
        self.sets = []
        self.rules = []
        self.result_set = None
        

    def add_set(self, name, low_limits, medium_limits, high_limits):
        self.sets.append({
            "name": name,
            "low_limits": low_limits,
            "medium_limits": medium_limits,
            "high_limits": high_limits
        })

    def add_result_set(self, name, low_limits, medium_limits, high_limits):
        self.result_set = {
            "name": name,
            "low_limits": low_limits,
            "medium_limits": medium_limits,
            "high_limits": high_limits
        }

    def discretize_sets(self, points):
        for group in self.sets:
            superior_limit = group["high_limits"][1]
            inferior_limit = group["low_limits"][0]
            interval = (superior_limit - inferior_limit)/(points - 1)
            values_list = []
            low_membership = []
            medium_membership = []
            high_membership = []
            for i in range(points):
                point_value = inferior_limit + (i * interval)
                values_list.append(point_value)
                low_membership.append(self.create_membership_groups(group, point_value, 'low'))
                medium_membership.append(self.create_membership_groups(group, point_value, 'medium'))
                high_membership.append(self.create_membership_groups(group, point_value, 'high'))
            group['values_list'] = values_list
            group['low_membership'] = low_membership
            group['medium_membership'] = medium_membership
            group['high_membership'] = high_membership

    def create_membership_groups(self, group, point_value, group_type):
        value = 1
        superior_limit = group[f"{group_type}_limits"][1]
        inferior_limit = group[f"{group_type}_limits"][0]
        medium_limit = (superior_limit + inferior_limit)/2
        if(point_value < inferior_limit or point_value > superior_limit):
            return 0
        if(point_value < medium_limit):
            if(group_type == 'low'): return value
            value = (point_value - inferior_limit)/(medium_limit - inferior_limit)
            return value
        if(point_value > medium_limit):
            if(group_type == 'high'): return value
            value = (superior_limit - point_value)/(superior_limit - medium_limit)
            return value
        return value
        
    def calculate_membership(self, value, set_name):
        target_set = None
        for s in self.sets:
            if(s['name'] == set_name):
                target_set = s
        index = find_nearest_index(target_set['values_list'], value)
        memberships =  [
            target_set['low_membership'][index], 
            target_set['medium_membership'][index], 
            target_set['high_membership'][index]
        ]

        return memberships

    def create_rules(self, description, key, output):
        new_rule = {
            "key": key,
            "description": description,
            "output": output,
        }
        self.rules.append(new_rule)

    def get_activated_rules(self, membership_groups, values):
        temperature_membership = membership_groups[0]
        volume_membership = membership_groups[1]
        rule_keys = []
        min_values = []
        for i in range(len(temperature_membership)):
            if temperature_membership[i] == 0: continue
            for j in range(len(volume_membership)):
                if volume_membership[j] == 0: continue
                rule_keys.append(f'{i}{j}')
                local_min = np.array([temperature_membership[i], volume_membership[j]])
                min_values.append(local_min.min())
        for i in range(len(rule_keys)):
            rule = self.find_rule_by_key(rule_keys[i])
            print(values, "activated", rule, min_values[i])
        print(membership_groups)
        print(min_values.index(np.array(min_values).max()))
        

    def find_rule_by_key(self, key):
        for rule in self.rules:
            if rule["key"] == key: return rule

def find_nearest_index(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index

        
