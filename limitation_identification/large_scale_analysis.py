import pandas as pd
from ast import literal_eval as load

def contain_specific_section(l1):
    lower_case = [i.lower() for i in l1]
    section_all = " ".join(lower_case)
    list = ["abstract", "discussion", "limitation", "weakness", "discussions", "limitations", "weaknesses", "caveat", "shortcoming", "drawback"]
    for i in list:
        if i in section_all:
            return True
    return False


large_scale_prediction = pd.read_csv("large_scale_data_with_predictions.csv")
large_scale_prediction["sid"] = large_scale_prediction["sid"].apply(load)
large_scale_prediction["sid"] = large_scale_prediction["sid"].str[0]

all_data_with_sections_df = pd.read_csv("data/11988_rct_articles.csv")

all_data_with_sections_df["section"] = all_data_with_sections_df["section"].apply(load)
all_data_with_sections_df["contain_section"] = all_data_with_sections_df.section.apply(contain_specific_section)

all_data = pd.merge(all_data_with_sections_df, large_scale_prediction,  how='left', left_on=['pmcids','sid'], right_on = ['pmcids','sid'])

all_data = all_data[["pmcids", "sid", "sentence_x", "true", "prediction", "section", "contain_section"]]

all_data = all_data.rename(index={"sentence_x": "sentences"})

all_data["true"] = all_data["true"].apply(load)
all_data["true"] = all_data["true"].str[0]
all_data["prediction"] = all_data["prediction"].apply(load)
all_data["prediction"] = all_data["prediction"].str[0]

print("total number of samples: ", len(all_data))
print("number of limitation samples: ", len(all_data[all_data["prediction"]==1]))
print("number of limitation samples in specific section: ", len(all_data[all_data["contain_section"] == True][all_data["prediction"]==1]))

all_data[all_data["contain_section"] == True][all_data["prediction"]==1].to_csv("all_limitation_data.csv")
print("number of no section: ", len(set(all_data.pmcids.to_list())) - len(set(all_data[all_data["contain_section"] == True].pmcids.to_list())))
print("no section pmcids: ", set(all_data.pmcids.to_list()) - set(all_data[all_data["contain_section"] == True].pmcids.to_list()))
all_data[all_data["contain_section"] == False][all_data["prediction"]==1].to_csv("limitation_mentions_out_of_scope.csv")
