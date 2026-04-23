import matplotlib.pyplot as plt

layer_x = [1, 2, 3, 4]
layer_y = [0.231, 0.414, 1.183, 1.391]
layer_labels = ["L1", "L2", "L6", "L8"]

head_x = [1, 2, 3]
head_y = [0.588, 0.741, 1.286]
head_labels = ["H1", "H2", "H8"]

embd_x = [1, 2, 3]
embd_y = [0.515, 1.843, 4.488]
embd_labels = ["E64", "E256", "E512"]

plt.figure(figsize=(8, 5))

plt.plot(layer_x, layer_y, marker='o', label='N_LAYER')
plt.plot(head_x, head_y, marker='o', label='N_HEAD')
plt.plot(embd_x, embd_y, marker='o', label='N_EMBD')

# N_LAYER offsets: step1=below, step2=below, step3=below, step4=right
layer_offsets = [(-12, -14), (-12, -14), (-12, -14), (10, 0)]
for x, y, label, off in zip(layer_x, layer_y, layer_labels, layer_offsets):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=off, ha='center', fontsize=9)

# N_HEAD offsets: step1=above-right, step2=above, step3=above
head_offsets = [(12, 6), (12, 6), (12, -14)]
for x, y, label, off in zip(head_x, head_y, head_labels, head_offsets):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=off, ha='center', fontsize=9)

# N_EMBD offsets: step1=above-left, step2=above, step3=above
embd_offsets = [(-12, 6), (0, 8), (0, 8)]
for x, y, label, off in zip(embd_x, embd_y, embd_labels, embd_offsets):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=off, ha='center', fontsize=9)

plt.xlabel('Sweep step')
plt.ylabel('Emissions (gCO$_2$eq)')
plt.title('Operational emissions across training parameter sweeps')
plt.xticks([1, 2, 3, 4])
plt.legend()
plt.tight_layout()
plt.savefig('codecarbon_sweeps.png', dpi=300, bbox_inches='tight')
plt.show()
