from fuzzy import Fuzzy

def main():
    inputs = [
        [965, 11],
        [920, 7.6],
        [1050, 6.3],
        [843, 8.6],
        [1122, 5.2],
    ]
    fuzzy = Fuzzy()
    fuzzy.add_set('temperature', [800, 1000], [900, 1100], [1000, 1200])
    fuzzy.add_set('volume', [2.0, 7.0], [4.5, 9.5], [7.0, 12.0])
    fuzzy.add_result_set('pressure', [4.0, 8.0], [6.0, 10.0], [8.0, 12.0])
    add_rules(fuzzy)
    fuzzy.discretize_sets(500)
    for element in inputs:
        temperature_value = element[0]
        volume_value = element[1]
        temperature_membership = fuzzy.calculate_membership(temperature_value, 'temperature')
        volume_membership = fuzzy.calculate_membership(volume_value, 'volume')
        fuzzy.get_activated_rules([temperature_membership, volume_membership], [temperature_value, volume_value])

def add_rules(fuzzy):
    fuzzy.create_rules('Temperatura Baixa e Volume Pequeno', '00', 'low')
    fuzzy.create_rules('Temperatura Média e Volume Pequeno', '10', 'low')
    fuzzy.create_rules('Temperatura Alta e Volume Pequeno', '20', 'medium')
    fuzzy.create_rules('Temperatura Baixa e Volume Médio', '01', 'low')
    fuzzy.create_rules('Temperatura Média e Volume Médio', '11', 'medium')
    fuzzy.create_rules('Temperatura Alta e Volume Médio', '21', 'high')
    fuzzy.create_rules('Temperatura Baixa e Volume Grande', '02', 'medium')
    fuzzy.create_rules('Temperatura Média e Volume Grande', '12', 'high')
    fuzzy.create_rules('Temperatura Alta e Volume Grande', '22', 'high')
    


main()