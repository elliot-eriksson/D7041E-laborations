# import modules & set up logging
import time
import gensim, logging, numpy as np
import help_functions as hf
import nltk

#@author: The first version of this code is the courtesy of Vadim Selyanik


nltk.download('wordnet')  # download the WordNet database
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
lemmatizer = nltk.WordNetLemmatizer()  # create a lemmatizer

# Load sentences from the lemmatized text file
sentences = []
file = open("lemmatized.text", "r")
for line in file:  # read the file and create list which contains all sentences found in the text
    sentences.append(line.split())
file.close()

# Parameters for Word2Vec training
dimensions = [10, 250, 1000]  # Different dimensionalities to test
threshold = 0.00055  # Threshold for the sampling of the words
num_of_simulations = 5  # Number of simulations to run
results = {}
times = []
accuracies = []
# Running simulations for each dimensionality
for dimension in dimensions:
    accuracies_per_dimension = []
    times_per_dimension = []
    for sim in range(num_of_simulations):
        start_time = time.time()
        # Train Word2Vec model with the given dimensionality
        model = gensim.models.Word2Vec(sentences, min_count=1, sample=threshold, sg=1, vector_size=dimension)  # create model using Word2Vec with the given parameters

        # The rest implements passing TOEFL tests
        number_of_tests = 80
        text_file = open('new_toefl.txt', 'r')
        right_answers = 0  # variable for correct answers
        number_skipped_tests = 0  # some tests could be skipped if there are no corresponding words in the vocabulary
        for i in range(number_of_tests):
            line = text_file.readline()  # read line in the file
            words = line.split()  # extract words from the line
            try:
                words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, 'v'), 'n'), 'a') for word in words]  # lemmatize words in the current test
                vectors = []
                if words[0] in model.wv:  # check if there is embedding for the query word
                    vectors.append(model.wv[words[0]])
                    for k in range(1, 5):
                        if words[k] in model.wv:  # if alternative has the embedding
                            vectors.append(model.wv[words[k]])  # assign the learned vector
                        else:
                            vectors.append(np.random.randn(dimension))  # assign random vector
                    right_answers += hf.get_answer_mod(vectors)  # find the closest vector and check if it is the correct answer
            except KeyError:  # if there is no representation for the query vector then skip
                number_skipped_tests += 1
                print("skipped test: " + str(i) + "; Line: " + str(words))
            except IndexError:
                print(i)
                print(line)
                print(words)
                break
        text_file.close()
        accuracy = 100 * float(right_answers) / float(number_of_tests)  # get the percentage
        accuracies_per_dimension.append(accuracy)
        results[dimension] = accuracy
        logging.info(f"Dimension: {dimension}, Accuracy: {accuracy:.2f}%")
        times_per_dimension.append(time.time() - start_time)
    avg_time = np.mean(times_per_dimension)
    avg_accuracy = np.mean(accuracies_per_dimension)
    accuracies.append(avg_accuracy)
    times.append(avg_time)
    

# Reporting the accuracy for all simulations
for dimension in dimensions:

    print(f"Dimension: {dimension}, Average Accuracy: {accuracies[dimensions.index(dimension)]:.2f}%, Average time: {times[dimensions.index(dimension)]:.2f}s")

