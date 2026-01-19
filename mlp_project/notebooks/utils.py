import matplotlib.pyplot as plt

def display(images):
    """Display a grid of images"""
    n = len(images)
    plt.figure(figsize=(15, 3))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()
