from fuzzy import Fuzzy

def main():
    fuzzy = Fuzzy()
    fuzzy.add_set('temperature', [800, 1000], [900, 1100], [1000, 1200])
    fuzzy.add_set('volume', [2.0, 7.0], [4.5, 9.5], [7.0, 12.0])
    fuzzy.add_result_set('pressure', [4.0, 8.0], [6.0, 10.0], [8.0, 12.0])
    add_rules(fuzzy)
    fuzzy.discretize_sets(500)
    temperature_membership = fuzzy.calculate_membership(920, 'temperature')
    volume_membership = fuzzy.calculate_membership(7.6, 'volume')
    fuzzy.get_activated_rules([temperature_membership, volume_membership], [920, 7.6])

def add_rules(fuzzy):
    fuzzy.create_rules('Temperatura Baixa e Volume Pequeno', '00', '0')
    fuzzy.create_rules('Temperatura Média e Volume Pequeno', '10', '0')
    fuzzy.create_rules('Temperatura Alta e Volume Pequeno', '20', '1')
    fuzzy.create_rules('Temperatura Baixa e Volume Médio', '01', '0')
    fuzzy.create_rules('Temperatura Média e Volume Médio', '11', '1')
    fuzzy.create_rules('Temperatura Alta e Volume Médio', '21', '2')
    fuzzy.create_rules('Temperatura Baixa e Volume Grande', '02', '1')
    fuzzy.create_rules('Temperatura Média e Volume Grande', '12', '2')
    fuzzy.create_rules('Temperatura Alta e Volume Grande', '22', '2')
    


main()