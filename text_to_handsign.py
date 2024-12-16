import os
import cv2
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Path to the folder containing hand sign images
IMAGE_FOLDER = './text_to_handsign'

def resize_and_pad(img, target_width, target_height):
    h, w, _ = img.shape
    scale = min(target_width / w, target_height / h)
    resized_w = int(w * scale)
    resized_h = int(h * scale)

    resized_img = cv2.resize(img, (resized_w, resized_h))
    padded_img = cv2.copyMakeBorder(
        resized_img,
        (target_height - resized_h) // 2,
        (target_height - resized_h + 1) // 2,
        (target_width - resized_w) // 2,
        (target_width - resized_w + 1) // 2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return padded_img

# Set of stop words (e.g., "is", "the", "and")
STOP_WORDS = {"is", "the", "and", "a", "an", "to", "of", "for", "on", "in", "at","are"}


def extract_keywords(text):
    words = word_tokenize(text.lower())
    keywords = [word for word in words if word not in STOP_WORDS and word.isalnum()]
    return keywords

def display_hand_signs(keywords):
    for keyword in keywords:
        image_path = os.path.join(IMAGE_FOLDER, f"{keyword}.jpg")  # Image file assumed to be in .jpg format
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            img = resize_and_pad(img, 1280, 720)
            cv2.imshow('Hand Sign', img)
            print(f"Displaying image for: {keyword}")
            cv2.waitKey(2000)  # Display each image for 2 seconds
        else:
            print(f"No image found for: {keyword}")
    cv2.destroyAllWindows()

def main():
    # Input text
    text = "What is your name?"

    # Extract keywords
    keywords = extract_keywords(text)
    print("Extracted Keywords:", keywords)

    # Display corresponding hand sign images
    display_hand_signs(keywords)

if __name__ == "__main__":
    main()
