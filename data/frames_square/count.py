import os

counter = 0
for folder in os.listdir('./'):
    if os.path.isdir(folder):
        print(folder)
        for img in os.listdir(os.path.join(folder, 'color')):
            counter += 1
print(counter)
