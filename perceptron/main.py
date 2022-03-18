from perceptron import Perceptron

training_dataset = []
test_dataset = []


def load_data():
    training_file_path = './perceptron/datasets/treinamento.txt'
    test_file_path = './perceptron/datasets/teste.txt'
    training_file_data = open(training_file_path, 'r')
    test_file_data = open(test_file_path, 'r')

    for line in training_file_data.readlines():
        element_array = []
        for element in line.split(' '):
            element_number = float(element.strip())
            element_array.append(element_number)
        training_dataset.append(element_array)

    for line in test_file_data.readlines():
        element_array = []
        for element in line.split(' '):
            element_number = float(element.strip())
            element_array.append(element_number)
        test_dataset.append(element_array)


def main():
    load_data()
    perceptron = Perceptron()
    alltests = []
    test_results = []
    for _ in range(5):
        perceptron.train(training_dataset)

    for element in test_dataset:
        result = perceptron.test(element)
        test_results.append(result)
    alltests.append(test_results)
    print(alltests)


main()
