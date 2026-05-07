import matplotlib.pyplot as plt

length_x = [1, 2, 3, 4, 5]
length_y = [0.009907, float("nan"), 0.002517, 0.002740, 0.004136]
length_labels = ["S50", "MS", "M200", "MH", "L400"]

temp_x = [1, 2, 3, 4, 5]
temp_y = [0.001662, 0.002503, 0.001597, 0.002463, 0.001619]
temp_labels = ["Tlow", "Tmidlow", "Tmid", "Tmidhigh", "Thigh"]

plt.figure(figsize=(8, 5))

plt.plot(length_x, length_y, marker="o", label="OUTPUT_LENGTH")
plt.plot(temp_x, temp_y, marker="o", label="TEMPERATURE")

# OUTPUT_LENGTH offsets: step1=above-left, step2=below, steps3-5=above
length_offsets = [(-12, 6), (0, -14), (0, 8), (0, 8), (0, 8)]
for x, y, label, off in zip(length_x, length_y, length_labels, length_offsets):
    plt.annotate(
        label, (x, y), textcoords="offset points", xytext=off, ha="center", fontsize=9
    )

# TEMPERATURE offsets: step1=below-left, steps2-4=above, step5=above-right
temp_offsets = [(-12, -14), (0, 8), (0, 8), (0, 8), (12, 6)]
for x, y, label, off in zip(temp_x, temp_y, temp_labels, temp_offsets):
    plt.annotate(
        label, (x, y), textcoords="offset points", xytext=off, ha="center", fontsize=9
    )

plt.xlabel("Sweep step")
plt.ylabel("Emissions (gCO$_2$eq)")
plt.title("Operational emissions across prompting parameter sweeps")
plt.xticks([1, 2, 3, 4, 5])
plt.legend()
plt.tight_layout()
plt.savefig("codecarbon_prompt_sweeps.png", dpi=300, bbox_inches="tight")
plt.show()
