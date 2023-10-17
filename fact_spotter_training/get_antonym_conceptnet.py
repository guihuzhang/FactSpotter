import csv
import json


def get_antonyms(csv_file):
    tmp_antonyms = {}
    tmp_synonyms = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            # 提取词
            word1 = row[2].split('/')[3].replace("_", " ")
            word2 = row[3].split('/')[3].replace("_", " ")
            if row[2].startswith("/c/en") and row[3].startswith("/c/en"):
                if row[1] == '/r/Antonym':
                    if word1 not in tmp_antonyms:
                        tmp_antonyms[word1] = [word2]
                    elif word2 not in tmp_antonyms[word1]:
                        tmp_antonyms[word1].append(word2)
                    if word2 not in tmp_antonyms:
                        tmp_antonyms[word2] = [word1]
                    elif word1 not in tmp_antonyms[word2]:
                        tmp_antonyms[word2].append(word1)
                elif row[1] == '/r/Synonym':
                    if word1 not in tmp_synonyms:
                        tmp_synonyms[word1] = [word2]
                    elif word2 not in tmp_synonyms[word1]:
                        tmp_synonyms[word1].append(word2)
                    if word2 not in tmp_synonyms:
                        tmp_synonyms[word2] = [word1]
                    elif word1 not in tmp_synonyms[word2]:
                        tmp_synonyms[word2].append(word1)
    return tmp_antonyms, tmp_synonyms


antonyms, synonyms = get_antonyms('conceptnet-assertions-5.7.0.csv')  # 替换为你的 CSV 文件名

with open('antonyms.json', 'w') as json_file:
    json.dump(antonyms, json_file, indent=4)

with open('synonyms.json', 'w') as json_file:
    json.dump(synonyms, json_file, indent=4)
