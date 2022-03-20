from adaline import Adaline

training_dataset = []
test_dataset = []


def load_data():
    training_file_path = './datasets/training.txt'
    test_file_path = './datasets/test.txt'
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
    adaline = Adaline()
    alltests = []
    test_results = []
    for _ in range(5):
        adaline.train(training_dataset)

    for _ in range(5):
        for element in test_dataset:
            result = adaline.test(element)
            test_results.append(result)
        alltests.append(test_results)
        test_results = []
    print(alltests)


main()
