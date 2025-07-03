
import json

def handle_lean4():
    all_data = []
    # Open the file and load its contents
    file_path = '/opt/tiger/CLIP/theorem_proving/miniF2F-lean4/minif2f_lean4.jsonl'
    test_data = []
    valid_data = []
    with open(file_path, "r") as f:
        for  line in f.readlines():
            data = json.loads(line)
            if data['split'] == "test":
                test_data.append(data)
            elif data['split'] == "valid":
                valid_data.append(data)
    save_path = '/opt/tiger/CLIP/theorem_proving/miniF2F-lean4/handle_minif2f_test.jsonl'
    with open(save_path, 'w') as f:
        json.dump(test_data, f, indent=4)

    save_path = '/opt/tiger/CLIP/theorem_proving/miniF2F-lean4/handle_minif2f_valid.jsonl'
    with open(save_path, 'w') as f:
        json.dump(valid_data, f, indent=4)



if __name__ == "__main__":
    handle_lean4()
