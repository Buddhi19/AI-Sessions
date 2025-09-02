import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys

    main_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    sys.path.append(main_dir)

    import marimo as mo
    import polars as pl
    import numpy as np
    import cv2 as cv
    import matplotlib.pyplot as plt
    return cv, main_dir, mo, np, os, plt


@app.cell
def _(main_dir, mo, os):
    src_example = os.path.join(
        main_dir, "2025-09-01-Kurunagala-Sri-Lanka","public","cnn.png"
    )
    mo.image(src_example)
    return


@app.cell
def _(mo):
    mo.md(r"""# What is Convolution""")
    return


@app.cell
def _(main_dir, mo, os):
    src_gif = os.path.join(
        main_dir, "2025-09-01-Kurunagala-Sri-Lanka","public","cnn.gif.png"
    )
    mo.image(src_gif)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # It's all simple MATHS

    $$
    \begin{equation}
    (1\times 1) + (1\times0) + (1\times-1) + (1\times 1) + (3\times0) + (4\times-1) + (1\times 1) + (4\times0) + (2\times-1) = -4
    \end{equation}
    $$

    # We continue with this to get,
    """
    )
    return


@app.cell
def _(main_dir, mo, os):
    src_gif_rerun = os.path.join(
        main_dir, "2025-09-01-Kurunagala-Sri-Lanka","public","cnn.gif"
    )
    mo.image(src_gif_rerun)
    return


@app.cell
def _(mo):
    mo.md(r"""# Okay now! What is the use if this?""")
    return


@app.cell
def _(cv, main_dir, mo, os):
    cat_png = os.path.join(
        main_dir, "2025-09-01-Kurunagala-Sri-Lanka","public","cat.png"
    )
    img = cv.imread(cat_png, cv.IMREAD_GRAYSCALE)

    cv.imwrite("cat_bw.png", img)
    cat_bw_png = os.path.join(
        main_dir, "2025-09-01-Kurunagala-Sri-Lanka","cat_bw.png"
    )
    mo.hstack([
            mo.vstack([mo.md("**Original**"), mo.image(cat_png, width=300)]),
            mo.vstack([mo.md("**Grayscale**"), mo.image(cat_bw_png, width=300)])
    ])
    return (img,)


@app.cell
def _(mo, np):
    average_kernel = np.ones((3, 3), np.float32) / 9
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)

    # Define custom kernel inputs
    custom_init = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    numbers = [
        [mo.ui.number(-10.0, 10.0, step=0.1, value=custom_init[i][j], label=f"[{i},{j}]") for j in range(3)]
        for i in range(3)
    ]

    # Kernel selection dropdown
    options = ["Average Blur", "Sharpen", "Sobel X", "Sobel Y", "Gaussian Blur", "Laplacian", "Edge Detection", "Custom"]
    selected = mo.ui.dropdown(options, value="Average Blur", label="")

    # Style the UI with HTML and CSS, escaping curly braces
    html_template = """
    <style>
      .kernel-container {{
        font-family: Arial, sans-serif;
        padding: 20px;
        background-color: #f5f5f5;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        max-width: 400px;
        margin: 10px auto;
      }}
      .dropdown-label {{
        font-size: 18px;
        font-weight: bold;
        color: #333;
        margin-bottom: 10px;
      }}
      .custom-kernel {{
        margin-top: 20px;
        padding: 10px;
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 5px;
      }}
      .custom-kernel-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        justify-items: center;
      }}
      .custom-kernel-title {{
        font-size: 16px;
        font-weight: bold;
        color: #444;
        margin-bottom: 10px;
      }}
      .marimo-ui-number input {{
        width: 70px;
        padding: 6px;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-size: 16px;
      }}
      .marimo-ui-number label {{
        font-size: 12px;
        color: #555;
      }}
      .marimo-ui-dropdown select {{
        padding: 10px;
        font-size: 16px;
        border-radius: 4px;
        border: 1px solid #ccc;
        background-color: #fff;
        width: 100%;
      }}
    </style>
    <div class="kernel-container">
      <div class="dropdown-label">Select Kernel</div>
      {dropdown}
      <div class="custom-kernel">
        <div class="custom-kernel-title">Custom Kernel (3x3)</div>
        <div class="custom-kernel-grid">
          {custom_inputs}
        </div>
      </div>
    </div>
    """
    # Generate custom inputs for 3x3 grid
    custom_inputs = "".join([
        f'<div>{numbers[i][j]}</div>' 
        for i in range(3) 
        for j in range(3)
    ])
    ui_html = mo.md(html_template.format(dropdown=selected, custom_inputs=custom_inputs))
    return (
        average_kernel,
        edge_kernel,
        gaussian_kernel,
        laplacian_kernel,
        numbers,
        selected,
        sharpen_kernel,
        sobel_x_kernel,
        sobel_y_kernel,
        ui_html,
    )


@app.cell
def _(
    average_kernel,
    cv,
    edge_kernel,
    gaussian_kernel,
    img,
    laplacian_kernel,
    mo,
    np,
    numbers,
    plt,
    selected,
    sharpen_kernel,
    sobel_x_kernel,
    sobel_y_kernel,
    ui_html,
):
    # Format kernel values as a 3x3 grid with larger text
    def format_kernel(kernel):
        return mo.md("""
    <style>
      .kernel-display {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        justify-items: center;
        font-size: 18px;
        font-family: monospace;
        margin: 10px 0;
        padding: 10px;
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 5px;
      }}
    </style>
    <div class="kernel-display">
      {kernel_values}
    </div>
    """.format(
        kernel_values="".join([
            f'<div>{kernel[i][j]:.2f}</div>'
            for i in range(3)
            for j in range(3)
        ])
    ))

    # Determine UI visibility and kernel
    if selected.value != "Custom":
        style_override = mo.md("""
    <style>
      .custom-kernel { display: none; }
    </style>
    """)
        controls = [style_override, ui_html]
    else:
        controls = [ui_html]

    if selected.value == "Custom":
        kernel = np.array([[n.value for n in row] for row in numbers], np.float32)
    elif selected.value == "Average Blur":
        kernel = average_kernel
    elif selected.value == "Sharpen":
        kernel = sharpen_kernel
    elif selected.value == "Sobel X":
        kernel = sobel_x_kernel
    elif selected.value == "Sobel Y":
        kernel = sobel_y_kernel
    elif selected.value == "Gaussian Blur":
        kernel = gaussian_kernel
    elif selected.value == "Laplacian":
        kernel = laplacian_kernel
    else:
        kernel = edge_kernel

    # Apply convolution
    output_img = cv.filter2D(img, -1, kernel)

    # Display kernel values
    kernel_display = format_kernel(kernel)

    # Visualize
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(img, cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("Original Image", fontsize=16)
    axs[1].imshow(output_img, cmap="gray")
    axs[1].axis("off")
    axs[1].set_title("Convolved Image", fontsize=16)
    plt.tight_layout()

    # Output controls, kernel display, and plot
    mo.vstack([
        mo.md(f"**Current Kernel: {selected.value}**"),
        kernel_display,
        *controls,
        fig
    ])
    return


if __name__ == "__main__":
    app.run()
