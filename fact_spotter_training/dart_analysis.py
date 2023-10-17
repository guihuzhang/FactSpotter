import json
from unidecode import unidecode


def webnlg_entity_format_correction(entity_letters):
    if "(" in entity_letters:
        entity_letters = entity_letters.split("(")[0]
    if "," in entity_letters:
        entity_letters = entity_letters.split(",")[0]
    return unidecode(entity_letters.replace("language", "").strip().lower())


def webnlg_sentences_format_correction(input_sentences):
    output_sent = []
    for each_sent in input_sentences:
        output_sent.append(unidecode(each_sent.lower()))
        # output_sent.append(unidecode(each_sent).replace(" ,", ",").replace(" .", ".").replace(" :", ":").replace(
        #     " )", ")").replace("( ", "(").replace(" 's", "'s").replace(" ;", ";").replace(" ' ", "' ").lower().strip())
    return output_sent


def phrase_in_sent_set(phrase, sent_set):
    for each_sent in sent_set:
        if phrase in each_sent.lower():
            return True
    return False


with open("../data/dart_new/val.json", 'r') as load_f:
    load_dict = json.load(load_f)
    all_constraints = []
    all_number = 0
    matched_num = 0
    prevent_pred = ["class name", "entity name", "domains", "literal name", "sentence category", "[title]",
                    "Sentence Data Source"]
    for each_record in load_dict:
        for each_triple in each_record["kbs"]:
            # print(each_record["kbs"][each_triple][0],
            #       phrase_in_sent_set(each_record["kbs"][each_triple][0],
            #                          each_record["text"]))
            # print(each_record["kbs"][each_triple][2][0][1],
            #       phrase_in_sent_set(each_record["kbs"][each_triple][2][0][1],
            #                          each_record["text"]))
            if each_record["kbs"][each_triple][2][0][0] not in prevent_pred:
                all_number += 1
                if phrase_in_sent_set(webnlg_entity_format_correction(each_record["kbs"][each_triple][0]),
                                      webnlg_sentences_format_correction(each_record["text"])):
                    matched_num += 1
                else:
                    print(each_record["kbs"][each_triple][0],
                          each_record["text"])
                for each_po in each_record["kbs"][each_triple][2]:
                    if each_po[0] != "status":
                        all_number += 1
                        if phrase_in_sent_set(webnlg_entity_format_correction(each_po[1]),
                                              webnlg_sentences_format_correction(each_record["text"])):
                            matched_num += 1
                        else:
                            print(each_po[1], each_record["text"])
    print(matched_num / all_number)
