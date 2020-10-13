import cv2
import numpy as np


def random_colors(N):
    # Ensure that the colors are consistent at each frame
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]

    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask in individual color channel.
    """
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )

    return image


def display_instances(image, boxes, masks, class_ids, class_names, scores=None, title=""):
    # Number of instances
    n_instances = boxes.shape[0]
    if not n_instances:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = random_colors(n_instances)

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    for i, color in enumerate(colors):
        # Bounding box
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]

        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)

        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = class_names[class_ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{}{:.2f}'.format(label, score) if score else label
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


def save_objects(image, boxes, class_ids, class_names):
    for cid in class_ids:
        for i, box in enumerate(boxes):
            print(box)
            if class_names[int(cid)] == 'bird':

                if not np.any(box):
                    continue

                # Bottom left corner to top right corner
                y1, x1, y2, x2 = box

                roi = image[x1:x2, y2:y1]

                with open(f'{i}.png', 'w') as fp:
                    pass
                if roi:
                    cv2.imwrite(f'{i}.png', roi)

                return roi
    return None
