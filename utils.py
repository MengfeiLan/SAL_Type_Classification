from collections import Counter

def read_sen_label_eda(file_path):
# Open the file in read mode
    sentences = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as file:
        file_contents = file.read()

    for line in file_contents.split("\n"):
        if len(line) > 2:
            sentences.append(line.split("\t")[1])
            labels.append(line.split("\t")[0])

    return sentences, labels

def convert_single_to_list(l1):
    if type(l1) is list:
        return l1
    else:
        return [l1]

def read_sen_label(file_path):
    with open(file_path, 'rb') as file:
        # Read the entire file into a string variable
        contents = file.read()

        try:
            decoded_string = contents.decode('utf-8')
        except UnicodeDecodeError:
            decoded_string = contents.decode('iso-8859-1')

        # Or read the file line by line

    sentences = []
    labels = []

    sen_label_pair = decoded_string.split("\n")

    for slp in sen_label_pair:
        new_sample = slp.strip("\r")
        new_sample = new_sample.split("\t")
        if len(new_sample) == 2:
            sentences.append(new_sample[0])
            labels.append(new_sample[1])

    return sentences, labels

def covert_list_to_one_hot(l1, num_label):
    onehot_all = []
    for i in l1:
      new_append = [0] * num_label
      for j in i:
          new_append[int(j)] = 1
      onehot_all.append(new_append)
    return onehot_all

def txt_to_dict(filename):
    with open(filename, 'r') as file:
        content = file.read()
        lines = content.splitlines()
        d = {}
        for line in lines:
            key, value = line.split('   ')
            d[key] = value
    return d

def select_top_similar_sentences(l1, l2, simcse_model):
    simcse_model.build_index(l1)
    results = simcse_model.search(l2)
    return results

def select_unique_labels(label_lists):
    all_labels = []
    for label_list in label_lists:
        all_labels.extend(label_list)
    return set(all_labels)

def select_specific_label(l1, label):
    if len(l1) > 1:
        return False
    elif l1[0] == label:
        return True
    else:
        return False

def flatten_list(l1):
    flatten_l = []
    for l in l1:
        flatten_l.extend(list(set(l)))
    return flatten_l

def describe_list_distribution(l1):
    flatten_l = []
    for l in l1:
        flatten_l.extend(list(set(l)))

    c = Counter(flatten_l)

    return c


def load_label_from_pretrained(file_path):
    return_dict = {}
    with open(file_path, 'r') as file:
        # Read the entire file
        contents = file.read()
        contents = contents.split("\n")

        # Read line by line
        for line in contents:
            line_split = line.split("\t")
            if len(line_split) > 1:
                return_dict[line_split[0]] = int(line_split[1])

    return return_dict

def load_thresholds_from_pretrained(file_path):
    return_dict = {}
    with open(file_path, 'r') as file:
        # Read the entire file
        contents = file.read()
        contents = contents.split("\n")

        # Read line by line
        for line in contents:
            line_split = line.split("\t")
            if len(line_split) > 1:
                return_dict[line_split[0]] = float(line_split[1])

    return return_dict

def convert_to_category_specific_list(categories, unique_labels):
    returned_vector = [0] * len(unique_labels)
    for i in categories:
        returned_vector[i] = 1

    return returned_vector

def go_back_to_origin(token_sequences):
    return_all = []
    for token_sequence in token_sequences:
        start = token_sequence.index('[PAD]')
        return_all.append(' '.join(token_sequence[1:start-1]))
    return return_all


def convert_id_to_label(results, id_to_labels):
    return_result = []
    for result in results:
        new_one = []
        for i in result:
            new_one.append(id_to_labels[i])
        return_result.append(new_one)
    return return_result
