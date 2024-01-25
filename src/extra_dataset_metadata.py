absolute_size_metadata = {
    "set_size": 3,
    "object_num": 1,
    "image_names": ["0.jpg", "1.jpg", "2.jpg"],
    "text_templates": {
        "0.jpg": [
            'the {obj} is big in relation to the image.',
            'the {obj} is large in relation to the image.',
            'the {obj} is relatively big within the image.',
            'the {obj} is relatively large within the image.',
        ],
        "1.jpg": [
            'the {obj} is medium-sized in relation to the image.',
            'the {obj} is moderate-sized in relation to the image.',
            'the {obj} is relatively moderate-sized within the image.',
            'the {obj} is relatively medium-sized within the image.',
        ],
        "2.jpg": [
            'the {obj} is small in relation to the image.',
            'the {obj} is tiny in relation to the image.',
            'the {obj} is relatively tiny within the image.',
            'the {obj} is relatively small within the image.',
        ]
    },
    "description": "`set_size` is the number of absolute size relationships, i.e. `big`, `same`, `small`"
}

relative_size_metadata = {
    "set_size": 3,
    "object_num": 2,
    "image_names": ["0.jpg", "1.jpg", "2.jpg"],
    "text_templates": {
        "0.jpg": [
            'the {sub} is smaller than the {obj} in size.',
            'the {sub} is smaller than the {obj} in scale.',
            'the {sub} is smaller in size/scale than the {obj}.',
            'the {sub} is on a smaller scale than the {obj}.',
            'the {sub} has a smaller size/scale compared to the {obj}'
        ],
        "1.jpg": [
            'the {sub} and the {obj} are of similar size.',
            'the {sub} and the {obj} are equally sized.',
            'the {sub} and the {obj} are the same size.',
            'the {sub} and the {obj} are of comparable size.',
            'the {sub} and the {obj} are in the same scale.',
            'the {sub} and the {obj} are of equivalent size.'
        ],
        "2.jpg": [
            'the {sub} is larger than the {obj} in size/scale.',
            'the {sub} is bigger than the {obj} in size.',
            'the {sub} is greater in size/scale than the {obj}.',
            'the {sub} is on a larger scale than the {obj}.',
            'the {sub} has a greater size/scale compared to the {obj}'
        ],
    }
}

absolute_spatial_metadata = {
    "set_size": 9,
    "object_num": 1,
    "image_names": ["0.jpg", "1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg"],
    "text_templates": {
        "0.jpg": [
            'the {obj} is located in the top left corner of the image.',
            'the {obj} is in the top-left corner of the image.',
            'the {obj} is situated in the upper-left corner of the image.',
            'the {obj} is at the top-left of the image.',
            'the {obj} is positioned in the top-left of the image.',
            'the {obj} is in the upper-left quadrant of the image.'
        ],
        "1.jpg": [
            'the {obj} is at the top of the image.',
            'the {obj} is positioned at the upper part of the image.',
            'the {obj} occupies the top portion of the image.'
        ],
        "2.jpg": [
            'the {obj} is located in the top right corner of the image.',
            'the {obj} is in the top-right corner of the image.',
            'the {obj} is situated in the upper-right corner of the image.',
            'the {obj} is at the top-right of the image.',
            'the {obj} is positioned in the top-right of the image.',
            'the {obj} is in the upper-right quadrant of the image.'
        ],
        "3.jpg": [
            'the {obj} is on the left of the image.',
            'the {obj} is to the left of the image.',
            'the {obj} is situated on the left side of the image.',
            'the {obj} is positioned to the left of the image.',
            'the {obj} is located on the left-hand side of the image.',
            'the {obj} occupies the left portion of the image.'
        ],
        "4.jpg": [
            'the {obj} is at the center of the image.',
            'the {obj} is in the middle of the image.',
            'the {obj} is positioned at the center of the image.',
            'the {obj} is situated in the middle of the image.',
            'the {obj} occupies the central position of the image.',
            'the {obj} is located centrally of the image.'
        ],
        "5.jpg": [
            'the {obj} is on the right of the image.',
            'the {obj} is to the right of the image.',
            'the {obj} is situated on the right side of the image.',
            'the {obj} is positioned to the right of the image.',
            'the {obj} is located on the right-hand side of the image.',
            'the {obj} occupies the right portion of the image.'
        ],
        "6.jpg": [
            'the {obj} is in the bottom-left corner of the image.',
            'the {obj} is located in the lower-left corner of the image.',
            'the {obj} is positioned at the bottom-left of the image.',
            'the {obj} is situated in the lower-left part of the image.'
        ],
        "7.jpg": [
            'the {obj} is located at the bottom of the image.',
            'the {obj} is positioned towards the lower part of the image.',
            'the {obj} is situated at the lower section of the image.',
            'the {obj} is placed in the lower portion of the image.',
            'the {obj} is at the lower end of the image.'
        ],
        "8.jpg": [
            'the {obj} is in the bottom-right corner of the image.',
            'the {obj} is located in the lower-right corner of the image.',
            'the {obj} is positioned at the bottom-right of the image.',
            'the {obj} is situated in the lower-right part of the image.'
        ],
    }
}

relative_spatial_metadata = {
    "set_size": 4,
    "object_num": 2,
    "image_names": ["0.jpg", "1.jpg", "2.jpg", "3.jpg"],
    "text_templates": {
        "0.jpg": [
            'the {sub} is to the left of the {obj}.',
            'the {sub} is on the left side of the {obj}.',
            'the {sub} is situated to the left of the {obj}.',
            'the {sub} is positioned on the left of the {obj}.'
        ],
        "1.jpg": [
            'the {sub} is above the {obj}.',
            'the {sub} is over the {obj}.',
            'the {sub} is on top of the {obj}.',
            'the {sub} is situated above the {obj}.',
            'the {sub} is positioned over the {obj}.'
        ],
        "2.jpg": [
            'the {sub} is to the right of the {obj}.',
            'the {sub} is on the right side of the {obj}.',
            'the {sub} is situated to the right of the {obj}.',
            'the {sub} is positioned on the right of the {obj}.'
        ],
        "3.jpg": [
            'the {sub} is below the {obj}.',
            'the {sub} is beneath the {obj}.',
            'the {sub} is under the {obj}.',
            'the {sub} is underneath the {obj}.',
            'the {sub} is positioned below the {obj}.',
            'the {sub} is placed beneath the {obj}.'
        ]
    }
}

count_metadata = {
    "set_size": 9,
    "object_num": 1,
    "image_names": ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg"],
    "text_templates": {
        "1.jpg": [
            'a picture of one {obj}.',
            'a photo of one {obj}.',
            'an image featuring one {obj}.',
            'a photograph capturing one {obj}.',
            'an illustration of one {obj}.',
            'a snapshot showcasing one {obj}.',
            'a visual of one {obj}.',
            'a view of one {obj}.',
            'a photo displaying one {obj}.'
        ],
        "2.jpg": [
            'a picture of two {obj}.',
            'a photo of two {obj}.',
            'an image featuring two {obj}.',
            'a photograph capturing two {obj}.',
            'an illustration of two {obj}.',
            'a snapshot showcasing two {obj}.',
            'a visual of two {obj}.',
            'a view of two {obj}.',
            'a photo displaying two {obj}.'
        ],
        "3.jpg": [
            'a picture of three {obj}.',
            'a photo of three {obj}.',
            'an image featuring three {obj}.',
            'a photograph capturing three {obj}.',
            'an illustration of three {obj}.',
            'a snapshot showcasing three {obj}.',
            'a visual of three {obj}.',
            'a view of three {obj}.',
            'a photo displaying three {obj}.'
        ],
        "4.jpg": [
            'a picture of four {obj}.',
            'a photo of four {obj}.',
            'an image featuring four {obj}.',
            'a photograph capturing four {obj}.',
            'an illustration of four {obj}.',
            'a snapshot showcasing four {obj}.',
            'a visual of four {obj}.',
            'a view of four {obj}.',
            'a photo displaying four {obj}.'
        ],
        "5.jpg": [
            'a picture of five {obj}.',
            'a photo of five {obj}.',
            'an image featuring five {obj}.',
            'a photograph capturing five {obj}.',
            'an illustration of five {obj}.',
            'a snapshot showcasing five {obj}.',
            'a visual of five {obj}.',
            'a view of five {obj}.',
            'a photo displaying five {obj}.'
        ],
        "6.jpg": [
            'a picture of six {obj}.',
            'a photo of six {obj}.',
            'an image featuring six {obj}.',
            'a photograph capturing six {obj}.',
            'an illustration of six {obj}.',
            'a snapshot showcasing six {obj}.',
            'a visual of six {obj}.',
            'a view of six {obj}.',
            'a photo displaying six {obj}.'
        ],
        "7.jpg": [
            'a picture of seven {obj}.',
            'a photo of seven {obj}.',
            'an image featuring seven {obj}.',
            'a photograph capturing seven {obj}.',
            'an illustration of seven {obj}.',
            'a snapshot showcasing seven {obj}.',
            'a visual of seven {obj}.',
            'a view of seven {obj}.',
            'a photo displaying seven {obj}.'
        ],
        "8.jpg": [
            'a picture of eight {obj}.',
            'a photo of eight {obj}.',
            'an image featuring eight {obj}.',
            'a photograph capturing eight {obj}.',
            'an illustration of eight {obj}.',
            'a snapshot showcasing eight {obj}.',
            'a visual of eight {obj}.',
            'a view of eight {obj}.',
            'a photo displaying eight {obj}.'
        ],
        "9.jpg": [
            'a picture of nine {obj}.',
            'a photo of nine {obj}.',
            'an image featuring nine {obj}.',
            'a photograph capturing nine {obj}.',
            'an illustration of nine {obj}.',
            'a snapshot showcasing nine {obj}.',
            'a visual of nine {obj}.',
            'a view of nine {obj}.',
            'a photo displaying nine {obj}.'
        ],
    }
}

existence_metadata = {
    "set_size": 2,
    "object_num": 1,
    "image_names": ["0.jpg", "1.jpg"],
    "text_templates": {
        "0.jpg": [
            "there is no {obj} in the image.",
            "a photo without a {obj}.",
            "the {obj} is absent from the image.",
            "the {obj} is not present in the image.",
            "the image dose not include {obj}.",
        ],
        "1.jpg": [
            "there is a {obj} in the image.",
            "a photo with a {obj}.",
            "a {obj} can be seen in the image.",
            "the image includes a {obj}.",
            "the {obj} is present in the image.",
        ]
    }
}
