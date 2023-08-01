import os

def read_token_list(path):
    with open(path, "r") as f:
        _token_list = f.readlines()
        token_list = [_[:-1] for _ in _token_list]
    return token_list
        
        
path = "../../data/depth/sample_tokens.txt"
sweeps_n = 100
token_list = read_token_list(path)
token_offset = 0

os.makedirs('output', exist_ok=True)
ct = 0
for sample_token in token_list:
    if ct <= token_offset:
        print("We skip token %d." % ct)
        ct += 1 
        continue
    print("Running token %d" % ct)
    cmd = "python -u ./scripts/run_pipeline.py --sample_token %s --sweeps_n %d" % (sample_token, sweeps_n)
    flag = os.system(cmd)
    if flag != 0:
        break
    os.system("mv ./output/6cam_depth_data ./output/token_%05d" % ct)
    ct += 1
    # import pdb; pdb.set_trace()