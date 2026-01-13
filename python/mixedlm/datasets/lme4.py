from __future__ import annotations

import pandas as pd


def load_sleepstudy() -> pd.DataFrame:
    """Load the sleepstudy dataset from lme4.

    Reaction times in a sleep deprivation study. On day 0 the subjects had
    their normal amount of sleep. Starting that night they were restricted
    to 3 hours of sleep per night. The observations represent the average
    reaction time on a series of tests given each day to each subject.

    Returns
    -------
    pd.DataFrame
        DataFrame with 180 observations and 3 columns:
        - Reaction: Average reaction time (ms)
        - Days: Number of days of sleep deprivation (0-9)
        - Subject: Subject identifier (308-372)

    Examples
    --------
    >>> from mixedlm.datasets import load_sleepstudy
    >>> sleepstudy = load_sleepstudy()
    >>> sleepstudy.head()
       Reaction  Days Subject
    0    249.56     0     308
    1    258.70     1     308
    2    250.80     2     308
    3    321.44     3     308
    4    356.85     4     308

    >>> # Fit a linear mixed model
    >>> from mixedlm import lmer
    >>> model = lmer("Reaction ~ Days + (Days | Subject)", data=sleepstudy)
    """
    data = {
        "Reaction": [
            249.5600, 258.7047, 250.8006, 321.4398, 356.8519,
            414.6901, 382.2038, 290.1486, 430.5853, 466.3535,
            222.7339, 205.2658, 202.9778, 204.7070, 207.7161,
            215.9618, 213.6303, 217.7272, 224.2957, 237.3142,
            199.0539, 194.3322, 234.3200, 232.8416, 229.3074,
            220.4579, 235.4208, 255.7511, 261.0125, 247.5153,
            321.5426, 300.4002, 283.8565, 285.1330, 285.7973,
            297.5855, 280.2396, 318.2613, 305.3495, 354.0487,
            287.6079, 285.0000, 301.8206, 320.1153, 316.2773,
            293.3187, 290.0750, 334.8177, 293.7469, 371.5811,
            234.8606, 242.8118, 272.9613, 309.7688, 317.4629,
            309.9976, 454.1619, 346.8311, 330.3003, 253.8644,
            283.8424, 289.5550, 276.7693, 299.8097, 297.1710,
            338.1665, 340.8485, 305.3211, 354.0032, 387.6167,
            265.4731, 276.2012, 243.3647, 254.6723, 279.0244,
            284.1912, 305.5248, 331.5229, 335.7469, 377.2990,
            241.6083, 273.9472, 254.4907, 270.8021, 251.4519,
            254.6362, 245.4523, 235.3110, 235.7541, 237.2466,
            312.3666, 313.8058, 291.6112, 346.1222, 365.7324,
            391.8385, 404.2601, 416.6923, 455.8643, 458.9167,
            236.1032, 230.3167, 238.9256, 254.9220, 250.7103,
            269.7744, 281.5648, 308.1020, 336.2806, 351.6451,
            256.2968, 243.4543, 256.2046, 255.5271, 268.9165,
            329.7247, 379.4445, 362.9184, 394.4872, 389.0527,
            250.5265, 300.0576, 269.8939, 280.5891, 271.8274,
            304.6336, 287.7466, 266.5955, 321.5418, 347.5655,
            221.6771, 298.1939, 326.8785, 346.8555, 348.7402,
            352.8287, 354.4266, 360.4326, 375.6406, 388.5417,
            271.9235, 268.4369, 257.2424, 277.6566, 314.8222,
            317.2135, 298.1353, 348.1229, 340.2800, 366.5131,
            225.2640, 234.5235, 238.9008, 240.4730, 267.5373,
            344.1937, 281.1481, 347.5855, 365.1630, 372.2288,
            269.8804, 272.4428, 277.8989, 281.7895, 279.1705,
            284.5120, 259.2658, 304.6306, 350.7807, 369.4692,
            269.4117, 273.4740, 297.5968, 310.6316, 287.1726,
            329.6076, 334.4818, 343.2199, 369.1417, 364.1236,
        ],
        "Days": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 18,
        "Subject": (
            ["308"] * 10 + ["309"] * 10 + ["310"] * 10 + ["330"] * 10 +
            ["331"] * 10 + ["332"] * 10 + ["333"] * 10 + ["334"] * 10 +
            ["335"] * 10 + ["337"] * 10 + ["349"] * 10 + ["350"] * 10 +
            ["351"] * 10 + ["352"] * 10 + ["369"] * 10 + ["370"] * 10 +
            ["371"] * 10 + ["372"] * 10
        ),
    }
    return pd.DataFrame(data)


def load_cbpp() -> pd.DataFrame:
    """Load the cbpp dataset from lme4.

    Contagious bovine pleuropneumonia (CBPP) data. This dataset describes
    the serological incidence of CBPP in zebu cattle during a follow-up
    survey implemented in 15 ings from Ethiopian herds.

    Returns
    -------
    pd.DataFrame
        DataFrame with 56 observations and 4 columns:
        - herd: Herd identifier (factor with 15 levels)
        - incidence: Number of new serological cases
        - size: Herd size at the beginning of the period
        - period: Time period (factor with 4 levels: 1-4)

    Examples
    --------
    >>> from mixedlm.datasets import load_cbpp
    >>> cbpp = load_cbpp()
    >>> cbpp.head()
      herd  incidence  size period
    0    1          2    14      1
    1    1          3    12      2
    2    1          4     9      3
    3    1          0     5      4
    4    2          3    22      1

    >>> # Fit a binomial GLMM
    >>> from mixedlm import glmer
    >>> from mixedlm.families import Binomial
    >>> model = glmer(
    ...     "incidence / size ~ period + (1 | herd)",
    ...     data=cbpp,
    ...     family=Binomial()
    ... )
    """
    data = {
        "herd": [
            "1", "1", "1", "1",
            "2", "2", "2", "2",
            "3", "3", "3", "3",
            "4", "4", "4", "4",
            "5", "5", "5", "5",
            "6", "6", "6", "6",
            "7", "7", "7", "7",
            "8", "8", "8", "8",
            "9", "9", "9", "9",
            "10", "10", "10", "10",
            "11", "11", "11", "11",
            "12", "12", "12", "12",
            "13", "13", "13", "13",
            "14", "14", "14", "14",
        ],
        "incidence": [
            2, 3, 4, 0,
            3, 1, 1, 0,
            4, 4, 3, 0,
            1, 2, 2, 0,
            3, 0, 2, 0,
            1, 3, 0, 0,
            6, 2, 0, 0,
            0, 2, 0, 0,
            2, 0, 0, 0,
            1, 2, 0, 0,
            0, 0, 1, 0,
            2, 2, 1, 0,
            0, 0, 1, 0,
            2, 2, 0, 0,
        ],
        "size": [
            14, 12, 9, 5,
            22, 18, 21, 22,
            17, 12, 9, 5,
            31, 28, 22, 16,
            10, 9, 9, 7,
            12, 10, 8, 3,
            30, 25, 24, 19,
            25, 24, 23, 22,
            9, 7, 5, 4,
            22, 18, 16, 12,
            13, 10, 9, 8,
            19, 17, 14, 8,
            8, 7, 5, 4,
            17, 15, 12, 10,
        ],
        "period": ["1", "2", "3", "4"] * 14,
    }
    return pd.DataFrame(data)
