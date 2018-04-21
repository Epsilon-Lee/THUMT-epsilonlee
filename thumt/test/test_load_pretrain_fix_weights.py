import tensorflow as tf

old_checkpoint = "train/left2right/eval"
new_checkpoint = "train/right2left/eval"

old_var_list = tf.train.list_variables(old_checkpoint)
new_var_list = tf.train.list_variables(new_checkpoint)

print("Old var list:")
print(old_var_list)
print("New var list:")
print(new_var_list)

visualize_var = "source_embedding"
real_name_old = ""
real_name_new = ""
# find name
for var_name, _ in old_var_list:
    if visualize_var in var_name:
        real_name_old = var_name
for var_name, _ in new_var_list:
    if visualize_var in var_name:
        real_name_new = var_name
old_reader = tf.train.load_checkpoint(old_checkpoint)
new_reader = tf.train.load_checkpoint(new_checkpoint)
old_tensor = old_reader.get_tensor(real_name_old)
new_tensor = old_reader.get_tensor(real_name_new)

print(old_tensor)
print(old_tensor.shape)
print(new_tensor)
print(new_tensor.shape)
print(old_tensor == new_tensor)
