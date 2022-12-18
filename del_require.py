with open("./requirements.txt", 'r', encoding='utf-8') as fr:
    with open("./filter_requirements.txt", 'w', encoding='utf-8') as fw:
        for line in fr:
            if ("home" not in line) and ("tmp" not in line):
                fw.write(line)