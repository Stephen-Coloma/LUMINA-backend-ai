import cv2
import numpy as np


def showImage(img, title='image', t=0, esc=False):
    if img is None:
        print(f"[Warning] Image is None, skipping display for: {title}")
        return

    cv2.imshow(title, img)
    if esc:
        while True:
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break
            # Exit loop if the window was closed manually
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    else:
        key = cv2.waitKey(t)

    # Only try to destroy if the window is still visible
    if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(title)


def class_colors(num_colors):
    class_colors = []
    for i in range(0, num_colors):
        hue = 255 * i / num_colors
        col = np.zeros((1, 1, 3)).astype("uint8")
        col[0][0][0] = hue
        col[0][0][1] = 128  # Saturation
        col[0][0][2] = 255  # Value
        cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
        col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
        class_colors.append(col)

    return class_colors


def roi2rect(img_name, img_np, img_data, label_list):
    colors = class_colors(len(label_list))
    for rect in img_data:
        bounding_box = [rect[0], rect[1], rect[2], rect[3]]
        xmin = int(bounding_box[0])
        ymin = int(bounding_box[1])
        xmax = int(bounding_box[2])
        ymax = int(bounding_box[3])
        pmin = (xmin, ymin)
        pmax = (xmax, ymax)

        label_array = rect[4:]
        # Find indices where label_array is 1
        indices = np.where(label_array == float(1))[0]

        if len(indices) > 0:
            # Get the first index where the label is 1
            index = int(indices[0])  # Use the first index
            label = label_list[index]

            color = colors[index]
            cv2.rectangle(img_np, pmin, pmax, color, 2)

            text_top = (xmin, ymin - 10)
            text_bot = (xmin + 80, ymin + 5)
            text_pos = (xmin + 5, ymin)
            cv2.rectangle(img_np, text_top, text_bot, colors[index], -1)
            cv2.putText(img_np, label, text_pos, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_PLAIN, 1, 1, 2)
        else:
            print(f"No label with value 1 found in rect: {rect}")

    # Display the image
    showImage(img=img_np, title=img_name)