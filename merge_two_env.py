

env1 = []
env1_key = []
with open('zipnerf.txt', 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 处理每一行的内容
        if '@' in line:
            continue
        env1.append(line)
        env1_key.append(line.split('==')[0])
env2_1 = []
with open('myenv.txt', 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 处理每一行的内容
        if line.split('==')[0] not in env1_key and '@' not in line:
            env2_1.append(line)



# 假设你有一个包含要写入的内容的列表
content_list = env2_1

# 打开文件以写入模式
with open('env2_1.txt', 'w') as file:
    # 遍历列表中的元素并写入文件
    for item in content_list:
        file.write(item + '\n')  # 写入元素并添加换行符

# print("内容已写入到output_file.txt文件中。")
