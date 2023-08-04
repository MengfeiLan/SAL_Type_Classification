from collections import Counter


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


def covert_list_to_one_hot(l1):
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

def describe_list_distribution(l1):
    flatten_l = []
    for l in l1:
        flatten_l.extend(list(set(l)))

    c = Counter(flatten_l)
    return c
