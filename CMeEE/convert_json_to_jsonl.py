import os
import json

def convert2jsonl():
    for file in os.listdir("CMeEE"):
        if not (file.startswith('CMeEE_') and file.endswith('.json')):
            continue
        path_input = os.path.join("CMeEE", file)
        with open(path_input, encoding='utf8') as f:
            data = json.load(f)
        lines = [json.dumps(i, ensure_ascii=False) for i in data]
        with open(path_input.replace('.json', '.jsonl'), encoding='utf8', mode='w') as f:
            f.write('\n'.join(lines))

if __name__ == '__main__':
    convert2jsonl()