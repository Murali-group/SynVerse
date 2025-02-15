import seaborn as sns
import matplotlib.pyplot as plt
color_palette = sns.cubehelix_palette(start=0.5, hue=1,
                                      gamma=0.4, dark=0.3, light=0.7,
                                      rot=-0.6, reverse=False, n_colors=3)
# Display the colors
sns.palplot(color_palette)
plt.show()