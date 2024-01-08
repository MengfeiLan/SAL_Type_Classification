import pandas as pd
from ast import literal_eval as load
from collections import Counter
import matplotlib.pyplot as plt
import colorsys
import re

def split_it(s):
    return re.search(r'tensor\((.*?), dtype=torch.int32\)', s).group(1)

def describe_list_distribution(l1):
    flatten_l = []
    for l in l1:
        flatten_l.extend(list(set(l)))

    c = Counter(flatten_l)

    print(c.items())
    return c.items()

def generate_light_colors(num_colors, lightness=0, saturation=10,hue_start=1.0, hue_end=0):
    colors = []
    for i in range(num_colors):
        saturation = i / (num_colors - 1)  # Vary saturation from 0 to 1
        hue = 0.56  # Hue value (0.5 is green, adjust as needed)
        value = 0.8  # Value (brightness) value (adjust as needed)

        # Convert HSV to RGB
        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)

        # Scale RGB values to 0-255 and convert to integers
        rgb_color = [int(val * 255) for val in rgb_color]

        # Convert RGB values to hexadecimal format
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)

        colors.append(hex_color)
    colors.reverse()
    return colors

def describe_paper_level_distribution(df):
    pmcids = list(set(df["pmcid"].to_list()))
    predictions = df["pred"].to_list()
    categories = list(set(i for prediction in predictions for i in prediction))
    paper_level_distribution_dict = {}
    for category in categories:
        paper_level_distribution_dict[category] = 0
    print("len(pmcids): ", len(pmcids))
    print("categories: ", categories)
    for pmcid in pmcids:
        df_sub = df[df["pmcid"] == pmcid]
        sub_preds = df_sub["pred"].to_list()
        sub_cat = list(set(pred_single for pred_singles in sub_preds for pred_single in pred_singles))
        for category in categories:
            if category in sub_cat:
                paper_level_distribution_dict[category] += 1
    return paper_level_distribution_dict

mode = "item"

df = pd.read_csv("checkpoint_1/coarse_promda_output_view_1_test.csv")
df["pmcid"] = df["pmcid"].apply(split_it)
len_pmcid = len(list(set(df["pmcid"].to_list())))
df["pred"]= df["pred"].apply(load)
if mode == "item":
    test_stat = describe_list_distribution(df.pred.to_list())
    print("test_stat: ", test_stat)
    overall_stat_coarse = {}
    for i in sorted(test_stat):
        if i[0] not in overall_stat_coarse:
            overall_stat_coarse[i[0]] = i[1]
        else:
            overall_stat_coarse[i[0]] += i[1]
    sorted_data = dict(sorted(overall_stat_coarse.items(), key=lambda item: item[1], reverse=True))

    # Extract keys (categories) and values (counts) from the dictionary
    categories = list(sorted_data.keys())
    counts = list(sorted_data.values())
    colors = generate_light_colors(len(categories))
    fig, ax = plt.subplots(figsize=(20, 7))

    total_count = len(df)
    percentages = [(count / total_count) * 100 for count in counts]

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    bars = plt.barh(categories, counts, color=colors)
    plt.gca().invert_yaxis()

    for bar, percentage in zip(bars, percentages):
        print(bar.get_width() + 0.5, " , ", bar.get_y() + bar.get_height() / 2)
        plt.text(bar.get_width() + 650, bar.get_y() + bar.get_height() / 2 + 0.2, f'{percentage:.1f}%', ha='center')

    plt.xlabel('Counts')
    plt.ylabel('Categories')
    plt.title('Bar Chart Example')

    # Show the chart
    plt.show()

elif mode == "document":
    test_stat_document = describe_paper_level_distribution(df)
    sorted_data = dict(sorted(test_stat_document.items(), key=lambda item: item[1], reverse=True))

    # Extract keys (categories) and values (counts) from the dictionary
    categories = list(sorted_data.keys())
    counts = list(sorted_data.values())
    colors = generate_light_colors(len(categories))
    fig, ax = plt.subplots(figsize=(20, 7))
    percentages = [(count / len_pmcid) * 100 for count in counts]

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    bars = plt.barh(categories, counts, color=colors)
    plt.gca().invert_yaxis()

    for bar, count, percentage in zip(bars, counts, percentages):
        plt.text(bar.get_width() + 650, bar.get_y() + bar.get_height() / 2 + 0.2, f'{count}, ' + f'{percentage:.1f}%', ha='center')

    plt.xlabel('Counts')
    plt.ylabel('Categories')
    plt.title('Bar Chart Example')

    # Show the chart
    plt.show()

